// Standalone RV1103 pipeline using opencv-mobile capture (which initializes
// the ISP through rkaiq) -> letterbox 320x320 -> RKNN INT8 zero-copy ->
// YOLO26 decode + NMS. Measures per-stage latency, FPS and peak RSS.
//
// usage: camera_pipeline MODEL.rknn [W H [FRAMES [DUMP_RAW]]]
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "rknn_api.h"

#include "decoder.h"

static const char *CLASS_NAMES[NCLASS] = {"plastic", "paper", "metal", "glass"};
static float material_threshold = 0.0f;  // opt-in naming policy; boxes/NMS are unchanged

static std::string material_label(const Detection &d) {
    if (unknown_material(d, material_threshold))
        return std::string("unknown (") + CLASS_NAMES[d.cls] + ")";
    return CLASS_NAMES[d.cls];
}

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static unsigned long peak_rss_kb() {
    FILE *fp = fopen("/proc/self/status", "r");
    char line[256];
    unsigned long value = 0;
    if (!fp) return 0;
    while (fgets(line, sizeof(line), fp))
        if (sscanf(line, "VmHWM: %lu kB", &value) == 1) break;
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
    data = (unsigned char *)malloc((size_t)length);
    if (!data || fread(data, 1, (size_t)length, fp) != (size_t)length) {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *size = (size_t)length;
    return data;
}

static const cv::Scalar CLASS_COLORS[NCLASS] = {
    cv::Scalar(0, 200, 0),    // plastic  (green, BGR)
    cv::Scalar(255, 160, 0),  // paper    (blue)
    cv::Scalar(0, 128, 255),  // metal    (orange)
    cv::Scalar(255, 0, 200),  // glass    (purple)
};

static void draw_detections(cv::Mat &img, const std::vector<Detection> &dets, float scale,
                            int pad_x, int pad_y) {
    for (const auto &d : dets) {
        int x1 = (int)((d.x1 - pad_x) / scale);
        int y1 = (int)((d.y1 - pad_y) / scale);
        int x2 = (int)((d.x2 - pad_x) / scale);
        int y2 = (int)((d.y2 - pad_y) / scale);
        cv::Scalar color = unknown_material(d, material_threshold) ? cv::Scalar(160, 160, 160) : CLASS_COLORS[d.cls];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", material_label(d).c_str(), d.score);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int ty = y1 - 4 < ts.height ? y1 + ts.height + 2 : y1 - 4;
        cv::rectangle(img, cv::Point(x1, ty - ts.height - 2),
                      cv::Point(x1 + ts.width + 4, ty + baseline), color, -1);
        cv::putText(img, label, cv::Point(x1 + 2, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

#define DOUBLE_CLICK_MS 350
#define LONG_PRESS_MS 1500
#define SCAN_INTERVAL_MS 2000
#define SCAN_DURATION_MS 30000
#define LED_NAME "work"

static void msleep(int ms) { usleep((useconds_t)ms * 1000); }

static void led_set(int on) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/leds/%s/brightness", LED_NAME);
    int fd = open(path, O_WRONLY);
    if (fd < 0) return;
    char c = on ? '1' : '0';
    if (write(fd, &c, 1) < 0) { /* ignore */ }
    close(fd);
}

static void led_release_trigger(void) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/leds/%s/trigger", LED_NAME);
    int fd = open(path, O_WRONLY);
    if (fd < 0) return;
    if (write(fd, "none", 4) < 0) { /* ignore */ }
    close(fd);
}

static void led_blink(int times, int on_ms, int off_ms) {
    for (int i = 0; i < times; i++) {
        led_set(1);
        msleep(on_ms);
        led_set(0);
        msleep(off_ms);
    }
}

static void fsync_path(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

static bool save_image(const std::string &path, const cv::Mat &img) {
    if (cv::imwrite(path, img)) return true;
    // Fallback: binary PPM, always available.
    std::string ppm = path;
    size_t dot = ppm.find_last_of('.');
    if (dot != std::string::npos) ppm = ppm.substr(0, dot);
    ppm += ".ppm";
    FILE *fp = fopen(ppm.c_str(), "wb");
    if (!fp) return false;
    fprintf(fp, "P6\n%d %d\n255\n", img.cols, img.rows);
    std::vector<unsigned char> buf;
    for (int y = 0; y < img.rows; y++) {
        const cv::Vec3b *row = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.cols; x++) {
            buf.push_back(row[x][2]);  // BGR -> RGB
            buf.push_back(row[x][1]);
            buf.push_back(row[x][0]);
        }
    }
    fwrite(buf.data(), 1, buf.size(), fp);
    fclose(fp);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s MODEL.rknn [W H [FRAMES [DUMP_RAW]]]\n"
                "       %s MODEL.rknn --shots OUTDIR INTERVAL_SEC TOTAL_SEC [W H]\n"
                "       %s MODEL.rknn --button OUTDIR [W H [MAX_SHOTS]]\n"
                "       %s MODEL.rknn --field OUTDIR [W H]\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    const char *setting = getenv("TVL_MATERIAL_MARGIN");
    if (setting) {
        char *end = NULL;
        errno = 0;
        material_threshold = strtof(setting, &end);
        if (end == setting || *end || errno || !std::isfinite(material_threshold) ||
            material_threshold < 0 || material_threshold > 1) {
            fprintf(stderr, "TVL_MATERIAL_MARGIN must be a number from 0 to 1 (0 disables abstention)\n");
            return 2;
        }
    }
    if (material_threshold > 0)
        printf("experimental unknown-material display: margin < %.4f; detected boxes are unchanged\n", material_threshold);
    const char *model_path = argv[1];
    bool shots_mode = argc > 2 && strcmp(argv[2], "--shots") == 0;
    bool button_mode = argc > 2 && strcmp(argv[2], "--button") == 0;
    bool field_mode = argc > 2 && strcmp(argv[2], "--field") == 0;
    const char *shots_dir = NULL;
    const char *button_dir = NULL;
    const char *field_dir = NULL;
    int button_max = 0;  // 0 = infinite
    double shots_interval = 5.0, shots_total = 120.0;
    int req_w = 640, req_h = 480;
    int frames = 60;
    const char *dump_raw = NULL;
    if (field_mode) {
        if (argc < 4) {
            fprintf(stderr, "field mode needs OUTDIR\n");
            return 2;
        }
        field_dir = argv[3];
        req_w = argc > 4 ? atoi(argv[4]) : 640;
        req_h = argc > 5 ? atoi(argv[5]) : 480;
    } else if (button_mode) {
        if (argc < 4) {
            fprintf(stderr, "button mode needs OUTDIR\n");
            return 2;
        }
        button_dir = argv[3];
        req_w = argc > 4 ? atoi(argv[4]) : 640;
        req_h = argc > 5 ? atoi(argv[5]) : 480;
        button_max = argc > 6 ? atoi(argv[6]) : 0;
    } else if (shots_mode) {
        if (argc < 6) {
            fprintf(stderr, "shots mode needs OUTDIR INTERVAL_SEC TOTAL_SEC\n");
            return 2;
        }
        shots_dir = argv[3];
        shots_interval = atof(argv[4]);
        shots_total = atof(argv[5]);
        req_w = argc > 6 ? atoi(argv[6]) : 640;
        req_h = argc > 7 ? atoi(argv[7]) : 480;
    } else {
        req_w = argc > 3 ? atoi(argv[2]) : 640;
        req_h = argc > 3 ? atoi(argv[3]) : 480;
        frames = argc > 4 ? atoi(argv[4]) : 60;
        dump_raw = argc > 5 ? argv[5] : NULL;
    }

    size_t model_size = 0;
    unsigned char *model = read_file(model_path, &model_size);
    if (!model) {
        fprintf(stderr, "read %s failed: %s\n", model_path, strerror(errno));
        return 1;
    }

    rknn_context ctx = 0;
    if (rknn_init(&ctx, model, (uint32_t)model_size, 0, NULL) != RKNN_SUCC) {
        fprintf(stderr, "rknn_init failed\n");
        return 1;
    }
    rknn_sdk_version ver;
    memset(&ver, 0, sizeof(ver));
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN_SUCC)
        printf("runtime=%s driver=%s\n", ver.api_version, ver.drv_version);

    rknn_input_output_num io;
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC || io.n_input != 1 || io.n_output != 1) {
        fprintf(stderr, "expected one input and one output tensor\n");
        return 1;
    }
    rknn_tensor_attr in_attr, out_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    if (rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_attr, sizeof(in_attr)) != RKNN_SUCC) {
        fprintf(stderr, "input tensor query failed\n");
        return 1;
    }
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    if (rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &out_attr, sizeof(out_attr)) != RKNN_SUCC) {
        fprintf(stderr, "output tensor query failed\n");
        return 1;
    }
    if (in_attr.n_dims != 4 || in_attr.dims[0] != 1 || in_attr.dims[1] != IMGSZ ||
        in_attr.dims[2] != IMGSZ || in_attr.dims[3] != 3 || in_attr.fmt != RKNN_TENSOR_NHWC ||
        in_attr.size != IMGSZ * IMGSZ * 3 || in_attr.size_with_stride < in_attr.size ||
        out_attr.n_dims != 3 || out_attr.dims[0] != 1 || out_attr.dims[1] != 8 || out_attr.dims[2] != 2100 ||
        out_attr.type != RKNN_TENSOR_INT8 || out_attr.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
        out_attr.size != 8 * 2100 || out_attr.size_with_stride < out_attr.size ||
        !std::isfinite(out_attr.scale) || out_attr.scale <= 0 || out_attr.zp < -128 || out_attr.zp > 127) {
        fprintf(stderr, "unsupported model tensor layout or affine quantization; verify the Nano320 export contract\n");
        return 1;
    }
    printf("output quantization: zero_point=%d scale=%.9g\n", out_attr.zp, out_attr.scale);
    printf("input dims=%ux%ux%ux%u type=%d size=%u\n", in_attr.dims[0], in_attr.dims[1],
           in_attr.dims[2], in_attr.dims[3], in_attr.type, in_attr.size);
    printf("output dims=%ux%ux%ux%u type=%d size=%u\n", out_attr.dims[0], out_attr.dims[1],
           out_attr.dims[2], out_attr.dims[3], out_attr.type, out_attr.size);

    in_attr.type = RKNN_TENSOR_UINT8;
    in_attr.fmt = RKNN_TENSOR_NHWC;
    rknn_tensor_mem *in_mem = rknn_create_mem(ctx, in_attr.size_with_stride);
    rknn_tensor_mem *out_mem = rknn_create_mem(ctx, out_attr.size_with_stride);
    if (!in_mem || !out_mem ||
        rknn_set_io_mem(ctx, in_mem, &in_attr) != RKNN_SUCC ||
        rknn_set_io_mem(ctx, out_mem, &out_attr) != RKNN_SUCC) {
        fprintf(stderr, "zero-copy setup failed\n");
        return 1;
    }

    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, req_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, req_h);
    if (!cap.open(0)) {
        fprintf(stderr, "camera open failed\n");
        return 1;
    }
    printf("requested capture=%dx%d\n", req_w, req_h);

    cv::Mat bgr, resized, padded, rgb;
    std::vector<Detection> dets;
    double t_cap = 0, t_pre = 0, t_npu = 0, t_post = 0, t_total = 0;
    int measured = 0, total_dets = 0, last_n = 0;

    if (field_mode) {
        mkdir(field_dir, 0755);
        std::string photos = std::string(field_dir) + "/photos";
        mkdir(photos.c_str(), 0755);
        const char *csv_name = material_threshold > 0 ? "/detections-material.csv" : "/detections.csv";
        FILE *csv = fopen((std::string(field_dir) + csv_name).c_str(), "a");
        if (csv && material_threshold > 0) {
            struct stat info;
            if (fstat(fileno(csv), &info) == 0 && info.st_size == 0)
                fprintf(csv, "uptime_ms,path,class,suggested_class,confidence,margin,threshold,x1,y1,x2,y2\n");
        }
        led_release_trigger();
        led_set(0);
        int evfd = open("/dev/input/event0", O_RDONLY | O_NONBLOCK);
        if (evfd < 0) {
            fprintf(stderr, "open /dev/input/event0: %s\n", strerror(errno));
            return 1;
        }

        auto take_photo = [&](const std::string &dir, const std::string &prefix, int idx) -> int {
            for (int k = 0; k < 3; k++) cap >> bgr;
            if (bgr.empty()) return -1;
            int w = bgr.cols, h = bgr.rows;
            float scale = std::min((float)IMGSZ / w, (float)IMGSZ / h);
            int rw = (int)(w * scale + 0.5f), rh = (int)(h * scale + 0.5f);
            float dw = (IMGSZ - rw) / 2.0f, dh = (IMGSZ - rh) / 2.0f;
            cv::resize(bgr, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
            int top = (int)(dh - 0.1f + 0.5f), bottom = (int)(dh + 0.1f + 0.5f);
            int left = (int)(dw - 0.1f + 0.5f), right = (int)(dw + 0.1f + 0.5f);
            cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                               cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
            memcpy(in_mem->virt_addr, rgb.data, (size_t)IMGSZ * IMGSZ * 3);
            if (rknn_run(ctx, NULL) != RKNN_SUCC) {
                fprintf(stderr, "rknn_run failed\n");
                return -1;
            }
            int n = decode_output((const int8_t *)out_mem->virt_addr, dets, out_attr.scale, out_attr.zp);
            cv::Mat annotated = bgr.clone();
            draw_detections(annotated, dets, scale, left, top);
            char path[256];
            snprintf(path, sizeof(path), "%s/%s_%04d.jpg", dir.c_str(), prefix.c_str(), idx);
            bool ok = save_image(path, annotated);
            if (ok) fsync_path(path);
            printf("%s/%s_%04d %s dets=%d", dir.c_str(), prefix.c_str(), idx,
                   ok ? "saved" : "SAVE_FAILED", n);
            for (const auto &dd : dets) printf(" [%s %.2f]", material_label(dd).c_str(), dd.score);
            printf("\n");
            fflush(stdout);
            if (csv) {
                double t = now_ms();
                for (const auto &dd : dets) {
                    float fx1 = (dd.x1 - left) / scale, fy1 = (dd.y1 - top) / scale;
                    float fx2 = (dd.x2 - left) / scale, fy2 = (dd.y2 - top) / scale;
                    if (material_threshold > 0) {
                        const char *label = unknown_material(dd, material_threshold) ? "unknown" : CLASS_NAMES[dd.cls];
                        fprintf(csv, "%.0f,%s/%s_%04d,%s,%s,%.4f,%.6f,%.6f,%.1f,%.1f,%.1f,%.1f\n",
                                t, dir.c_str(), prefix.c_str(), idx, label, CLASS_NAMES[dd.cls], dd.score,
                                dd.material_margin, material_threshold, fx1, fy1, fx2, fy2);
                    } else {
                        fprintf(csv, "%.0f,%s/%s_%04d,%s,%.4f,%.1f,%.1f,%.1f,%.1f\n", t, dir.c_str(),
                                prefix.c_str(), idx, CLASS_NAMES[dd.cls], dd.score, fx1, fy1, fx2, fy2);
                    }
                }
                fflush(csv);
            }
            return n;
        };

        printf("field mode: dir=%s (1x=photo, 2x=scan %ds every %ds, hold=stop)\n", field_dir,
               SCAN_DURATION_MS / 1000, SCAN_INTERVAL_MS / 1000);
        fflush(stdout);
        bool scanning = false;
        double scan_end = 0, next_cap = 0;
        int scan_idx = 0, photo_idx = 0, scan_no = 0;
        std::string scan_dir;
        int clicks = 0;
        double last_click = 0, down_t = 0;
        bool down = false;

        // Resume counters from existing files so a reboot never overwrites captures.
        {
            DIR *dp = opendir(photos.c_str());
            if (dp) {
                struct dirent *de;
                while ((de = readdir(dp))) {
                    int n = -1;
                    if (sscanf(de->d_name, "shot_%04d.jpg", &n) == 1 && n >= photo_idx)
                        photo_idx = n + 1;
                }
                closedir(dp);
            }
            dp = opendir(field_dir);
            if (dp) {
                struct dirent *de;
                while ((de = readdir(dp))) {
                    int n = -1;
                    if (sscanf(de->d_name, "scan_%03d", &n) == 1 && n >= scan_no)
                        scan_no = n;
                }
                closedir(dp);
            }
        }
        printf("field resume: photos from %d, scans from %d\n", photo_idx, scan_no + 1);
        fflush(stdout);

        while (true) {
            struct pollfd pfd;
            pfd.fd = evfd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            if (poll(&pfd, 1, 200) > 0) {
                struct input_event ev;
                while (read(evfd, &ev, sizeof(ev)) == (ssize_t)sizeof(ev)) {
                    if (ev.type != EV_KEY) continue;
                    if (ev.value == 1) {
                        down = true;
                        down_t = now_ms();
                    } else if (ev.value == 0 && down) {
                        down = false;
                        if (now_ms() - down_t >= LONG_PRESS_MS) {
                            if (scanning) {
                                scanning = false;
                                led_set(0);
                                led_blink(2, 60, 60);
                                printf("scan stopped\n");
                                fflush(stdout);
                            } else {
                                scan_no++;
                                char d[256];
                                snprintf(d, sizeof(d), "%s/scan_%03d", field_dir, scan_no);
                                mkdir(d, 0755);
                                scan_dir = d;
                                scan_idx = 0;
                                next_cap = now_ms();
                                scan_end = next_cap + SCAN_DURATION_MS;
                                scanning = true;
                                led_set(1);
                                led_blink(3, 60, 60);
                                printf("scan started: %s\n", scan_dir.c_str());
                                fflush(stdout);
                            }
                        } else {
                            clicks++;
                            last_click = now_ms();
                        }
                    }
                }
            }
            if (clicks > 0 && now_ms() - last_click > DOUBLE_CLICK_MS) {
                if (clicks == 1) {
                    led_blink(1, 50, 50);
                    take_photo(photos, "shot", photo_idx++);
                } else if (!scanning) {
                    scan_no++;
                    char d[256];
                    snprintf(d, sizeof(d), "%s/scan_%03d", field_dir, scan_no);
                    mkdir(d, 0755);
                    scan_dir = d;
                    scan_idx = 0;
                    next_cap = now_ms();
                    scan_end = next_cap + SCAN_DURATION_MS;
                    scanning = true;
                    led_set(1);
                    led_blink(3, 60, 60);
                    printf("scan started: %s\n", scan_dir.c_str());
                    fflush(stdout);
                }
                clicks = 0;
            }
            if (scanning) {
                double t = now_ms();
                if (t >= scan_end) {
                    scanning = false;
                    led_set(0);
                    led_blink(2, 60, 60);
                    printf("scan finished: %s\n", scan_dir.c_str());
                    fflush(stdout);
                } else if (t >= next_cap) {
                    take_photo(scan_dir, "scan", scan_idx++);
                    next_cap += SCAN_INTERVAL_MS;
                }
            }
        }
    }

    if (button_mode) {
        mkdir(button_dir, 0755);
        int evfd = open("/dev/input/event0", O_RDONLY | O_NONBLOCK);
        if (evfd < 0) {
            fprintf(stderr, "open /dev/input/event0: %s\n", strerror(errno));
            return 1;
        }
        printf("button mode: dir=%s max=%d (press board key to capture)\n", button_dir,
               button_max);
        fflush(stdout);
        int idx = 0;
        double last = 0;
        while (button_max == 0 || idx < button_max) {
            struct pollfd pfd;
            pfd.fd = evfd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            if (poll(&pfd, 1, 1000) <= 0) continue;
            struct input_event ev;
            while (read(evfd, &ev, sizeof(ev)) == (ssize_t)sizeof(ev)) {
                if (ev.type != EV_KEY || ev.value != 1) continue;
                double t = now_ms();
                if (t - last < 400) continue;
                last = t;
                for (int k = 0; k < 3; k++) cap >> bgr;
                if (bgr.empty()) continue;
                int w = bgr.cols, h = bgr.rows;
                float scale = std::min((float)IMGSZ / w, (float)IMGSZ / h);
                int rw = (int)(w * scale + 0.5f), rh = (int)(h * scale + 0.5f);
                float dw = (IMGSZ - rw) / 2.0f, dh = (IMGSZ - rh) / 2.0f;
                cv::resize(bgr, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
                int top = (int)(dh - 0.1f + 0.5f), bottom = (int)(dh + 0.1f + 0.5f);
                int left = (int)(dw - 0.1f + 0.5f), right = (int)(dw + 0.1f + 0.5f);
                cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                                   cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
                cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
                memcpy(in_mem->virt_addr, rgb.data, (size_t)IMGSZ * IMGSZ * 3);
                if (rknn_run(ctx, NULL) != RKNN_SUCC) {
                    fprintf(stderr, "rknn_run failed\n");
                    break;
                }
                int n = decode_output((const int8_t *)out_mem->virt_addr, dets, out_attr.scale, out_attr.zp);
                cv::Mat annotated = bgr.clone();
                draw_detections(annotated, dets, scale, left, top);
                char path[256];
                snprintf(path, sizeof(path), "%s/press_%04d.jpg", button_dir, idx);
                bool ok = save_image(path, annotated);
                printf("press %02d key=%d %s dets=%d", idx, ev.code, ok ? "saved" : "SAVE_FAILED",
                       n);
                for (const auto &d : dets) printf(" [%s %.2f]", material_label(d).c_str(), d.score);
                printf("\n");
                fflush(stdout);
                idx++;
            }
        }
        printf("button_done=%d peak_rss_kb=%lu\n", idx, peak_rss_kb());
        close(evfd);
        cap.release();
        rknn_destroy_mem(ctx, in_mem);
        rknn_destroy_mem(ctx, out_mem);
        rknn_destroy(ctx);
        free(model);
        return 0;
    }

    if (shots_mode) {
        mkdir(shots_dir, 0755);
        printf("shots mode: dir=%s interval=%.1fs total=%.1fs\n", shots_dir, shots_interval,
               shots_total);
        int shot = 0;
        double start = now_ms();
        double next = start;
        while (now_ms() - start < shots_total * 1000.0) {
            for (int k = 0; k < 3; k++) cap >> bgr;  // refresh stale buffers
            if (bgr.empty()) { usleep(100000); continue; }
            int w = bgr.cols, h = bgr.rows;
            float scale = std::min((float)IMGSZ / w, (float)IMGSZ / h);
            int rw = (int)(w * scale + 0.5f), rh = (int)(h * scale + 0.5f);
            float dw = (IMGSZ - rw) / 2.0f, dh = (IMGSZ - rh) / 2.0f;
            cv::resize(bgr, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
            int top = (int)(dh - 0.1f + 0.5f), bottom = (int)(dh + 0.1f + 0.5f);
            int left = (int)(dw - 0.1f + 0.5f), right = (int)(dw + 0.1f + 0.5f);
            cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                               cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
            memcpy(in_mem->virt_addr, rgb.data, (size_t)IMGSZ * IMGSZ * 3);
            if (rknn_run(ctx, NULL) != RKNN_SUCC) {
                fprintf(stderr, "rknn_run failed\n");
                break;
            }
            int n = decode_output((const int8_t *)out_mem->virt_addr, dets, out_attr.scale, out_attr.zp);
            cv::Mat annotated = bgr.clone();
            draw_detections(annotated, dets, scale, left, top);
            char path[256];
            snprintf(path, sizeof(path), "%s/shot_%03d.jpg", shots_dir, shot);
            bool ok = save_image(path, annotated);
            printf("shot %02d %s dets=%d", shot, ok ? "saved" : "SAVE_FAILED", n);
            for (const auto &d : dets) printf(" [%s %.2f]", material_label(d).c_str(), d.score);
            printf("\n");
            fflush(stdout);
            shot++;
            next += shots_interval * 1000.0;
            double wait = next - now_ms();
            if (wait > 0) usleep((useconds_t)(wait * 1000));
        }
        printf("shots_done=%d peak_rss_kb=%lu\n", shot, peak_rss_kb());
        cap.release();
        rknn_destroy_mem(ctx, in_mem);
        rknn_destroy_mem(ctx, out_mem);
        rknn_destroy(ctx);
        free(model);
        return 0;
    }

    for (int f = 0; f < frames + 5; f++) {
        double t0 = now_ms();
        cap >> bgr;
        if (bgr.empty()) {
            fprintf(stderr, "empty frame\n");
            break;
        }
        double t1 = now_ms();

        int w = bgr.cols, h = bgr.rows;
        float scale = std::min((float)IMGSZ / w, (float)IMGSZ / h);
        int rw = (int)(w * scale + 0.5f), rh = (int)(h * scale + 0.5f);
        float dw = (IMGSZ - rw) / 2.0f, dh = (IMGSZ - rh) / 2.0f;
        cv::resize(bgr, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
        int top = (int)(dh - 0.1f + 0.5f), bottom = (int)(dh + 0.1f + 0.5f);
        int left = (int)(dw - 0.1f + 0.5f), right = (int)(dw + 0.1f + 0.5f);
        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
        if (dump_raw && f == 5) {
            FILE *fp = fopen(dump_raw, "wb");
            if (fp) { fwrite(rgb.data, 1, (size_t)IMGSZ * IMGSZ * 3, fp); fclose(fp); }
        }
        memcpy(in_mem->virt_addr, rgb.data, (size_t)IMGSZ * IMGSZ * 3);
        double t2 = now_ms();
        if (rknn_run(ctx, NULL) != RKNN_SUCC) {
            fprintf(stderr, "rknn_run failed\n");
            break;
        }
        double t3 = now_ms();
        last_n = decode_output((const int8_t *)out_mem->virt_addr, dets, out_attr.scale, out_attr.zp);
        double t4 = now_ms();

        if (f >= 5) {
            t_cap += t1 - t0;
            t_pre += t2 - t1;
            t_npu += t3 - t2;
            t_post += t4 - t3;
            t_total += t4 - t0;
            measured++;
            total_dets += last_n;
        }
        if (f == 5) printf("frame_size=%dx%d scale=%.4f pad=%d,%d\n", w, h, scale, left, top);
    }
    printf("detections(last frame)=%d\n", last_n);
    for (const auto &d : dets) {
        float sx = 1.0f, sy = 1.0f;
        (void)sx; (void)sy;
        printf("  %-7s %.3f\n", material_label(d).c_str(), d.score);
    }
    if (measured) {
        printf("frames=%d avg_dets=%.2f\n", measured, (double)total_dets / measured);
        printf("latency_ms capture=%.2f preprocess=%.2f npu=%.2f postprocess=%.2f total=%.2f\n",
               t_cap / measured, t_pre / measured, t_npu / measured, t_post / measured,
               t_total / measured);
        printf("fps=%.2f\n", 1000.0 * measured / t_total);
    }
    printf("process_peak_rss_kb=%lu\n", peak_rss_kb());

    cap.release();
    rknn_destroy_mem(ctx, in_mem);
    rknn_destroy_mem(ctx, out_mem);
    rknn_destroy(ctx);
    free(model);
    return 0;
}
