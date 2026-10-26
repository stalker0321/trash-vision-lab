#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rknn_api.h"

static const char *type_name(rknn_tensor_type type) {
    switch (type) {
    case RKNN_TENSOR_FLOAT32: return "float32";
    case RKNN_TENSOR_FLOAT16: return "float16";
    case RKNN_TENSOR_INT8: return "int8";
    case RKNN_TENSOR_UINT8: return "uint8";
    case RKNN_TENSOR_INT16: return "int16";
    case RKNN_TENSOR_UINT16: return "uint16";
    case RKNN_TENSOR_INT32: return "int32";
    case RKNN_TENSOR_UINT32: return "uint32";
    case RKNN_TENSOR_INT64: return "int64";
    case RKNN_TENSOR_BOOL: return "bool";
    default: return "unknown";
    }
}

static const char *fmt_name(rknn_tensor_format fmt) {
    switch (fmt) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    default: return "undefined";
    }
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static unsigned long peak_rss_kb(void) {
    FILE *fp = fopen("/proc/self/status", "r");
    char line[256];
    unsigned long value = 0;
    if (!fp) return 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "VmHWM: %lu kB", &value) == 1) break;
    }
    fclose(fp);
    return value;
}

static unsigned char *read_file(const char *path, size_t *size) {
    FILE *fp = fopen(path, "rb");
    unsigned char *data;
    long length;
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0 || (length = ftell(fp)) <= 0 ||
        fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    data = malloc((size_t)length);
    if (!data || fread(data, 1, (size_t)length, fp) != (size_t)length) {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *size = (size_t)length;
    return data;
}

static void print_attr(const char *role, const rknn_tensor_attr *a) {
    uint32_t i;
    printf("%s[%u] name=%s dims=", role, a->index, a->name);
    for (i = 0; i < a->n_dims; ++i) printf("%s%u", i ? "x" : "", a->dims[i]);
    printf(" elems=%u size=%u fmt=%s type=%s qnt=%d zp=%d scale=%.9g\n",
           a->n_elems, a->size, fmt_name(a->fmt), type_name(a->type),
           a->qnt_type, a->zp, a->scale);
}

int main(int argc, char **argv) {
    const int timed_runs = 10;
    unsigned char *model = NULL;
    unsigned char *input_data = NULL;
    size_t model_size = 0;
    rknn_context ctx = 0;
    rknn_sdk_version version;
    rknn_input_output_num io;
    rknn_tensor_attr input_attr;
    rknn_tensor_attr *output_attrs = NULL;
    rknn_tensor_mem *input_mem = NULL;
    rknn_tensor_mem **output_mems = NULL;
    double total_ms = 0.0, min_ms = 1e30, max_ms = 0.0;
    int ret = 1, i;

    if (argc != 2 && argc != 4) {
        fprintf(stderr, "usage: %s MODEL.rknn [INPUT_RGB_320.raw OUTPUT_I8.raw]\n", argv[0]);
        return 2;
    }
    model = read_file(argv[1], &model_size);
    if (!model) {
        fprintf(stderr, "read %s failed: %s\n", argv[1], strerror(errno));
        goto out;
    }
    printf("model_bytes=%zu\n", model_size);
    ret = rknn_init(&ctx, model, (uint32_t)model_size, 0, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_init failed: %d\n", ret);
        goto out;
    }
    memset(&version, 0, sizeof(version));
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == RKNN_SUCC)
        printf("runtime=%s driver=%s\n", version.api_version, version.drv_version);
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC) {
        fprintf(stderr, "RKNN_QUERY_IN_OUT_NUM failed\n");
        goto out;
    }
    printf("inputs=%u outputs=%u\n", io.n_input, io.n_output);
    if (io.n_input != 1 || io.n_output == 0) {
        fprintf(stderr, "unexpected tensor counts\n");
        goto out;
    }
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    if (rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attr, sizeof(input_attr)) != RKNN_SUCC)
        goto out;
    print_attr("input", &input_attr);
    output_attrs = calloc(io.n_output, sizeof(*output_attrs));
    output_mems = calloc(io.n_output, sizeof(*output_mems));
    if (!output_attrs || !output_mems) goto out;
    for (i = 0; i < (int)io.n_output; ++i) {
        output_attrs[i].index = (uint32_t)i;
        if (rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,
                       &output_attrs[i], sizeof(output_attrs[i])) != RKNN_SUCC)
            goto out;
        print_attr("output", &output_attrs[i]);
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!output_mems[i] ||
            rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]) != RKNN_SUCC) {
            fprintf(stderr, "output[%d] zero-copy setup failed\n", i);
            goto out;
        }
    }

    input_attr.type = RKNN_TENSOR_UINT8;
    input_attr.fmt = RKNN_TENSOR_NHWC;
    input_mem = rknn_create_mem(ctx, input_attr.size_with_stride);
    if (!input_mem) goto out;
    memset(input_mem->virt_addr, 0, input_mem->size);
    if (argc == 4) {
        size_t input_size = 0;
        input_data = read_file(argv[2], &input_size);
        if (!input_data || input_size != input_attr.n_elems) {
            fprintf(stderr, "input must contain exactly %u bytes, got %zu\n",
                    input_attr.n_elems, input_size);
            goto out;
        }
        memcpy(input_mem->virt_addr, input_data, input_size);
    }
    if ((ret = rknn_set_io_mem(ctx, input_mem, &input_attr)) != RKNN_SUCC) {
        fprintf(stderr, "input zero-copy setup failed: %d\n", ret);
        goto out;
    }
    if ((ret = rknn_run(ctx, NULL)) != RKNN_SUCC) {
        fprintf(stderr, "warmup inference failed: %d\n", ret);
        goto out;
    }
    for (i = 0; i < timed_runs; ++i) {
        double start = monotonic_ms(), elapsed;
        if ((ret = rknn_run(ctx, NULL)) != RKNN_SUCC) {
            fprintf(stderr, "timed inference %d failed: %d\n", i, ret);
            goto out;
        }
        elapsed = monotonic_ms() - start;
        total_ms += elapsed;
        if (elapsed < min_ms) min_ms = elapsed;
        if (elapsed > max_ms) max_ms = elapsed;
    }
    printf("inference_ms mean=%.3f min=%.3f max=%.3f runs=%d\n",
           total_ms / timed_runs, min_ms, max_ms, timed_runs);
    printf("process_peak_rss_kb=%lu\n", peak_rss_kb());
    if (argc == 4) {
        FILE *fp = fopen(argv[3], "wb");
        size_t wrote;
        if (!fp) {
            fprintf(stderr, "open output failed: %s\n", strerror(errno));
            goto out;
        }
        wrote = fwrite(output_mems[0]->virt_addr, 1, output_attrs[0].size, fp);
        fclose(fp);
        if (wrote != output_attrs[0].size) {
            fprintf(stderr, "short output write: %zu/%u\n", wrote, output_attrs[0].size);
            goto out;
        }
        printf("output_file=%s bytes=%zu\n", argv[3], wrote);
    }
    ret = 0;
out:
    if (ctx && output_mems) {
        for (i = 0; i < (int)io.n_output; ++i)
            if (output_mems[i]) rknn_destroy_mem(ctx, output_mems[i]);
    }
    if (ctx && input_mem) rknn_destroy_mem(ctx, input_mem);
    if (ctx) rknn_destroy(ctx);
    free(output_mems);
    free(output_attrs);
    free(input_data);
    free(model);
    return ret;
}
