// Dense normalized 1x8x2100 INT8 output used by the Nano320 RV1103 export.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#define IMGSZ 320
#define CONF_THRES 0.25f
#define IOU_THRES 0.45f
#define MAX_DET 300
#define NCLASS 4

struct Detection {
    int cls;
    float score;
    float material_margin;
    float x1, y1, x2, y2;
};

static inline bool unknown_material(const Detection &d, float threshold) {
    return threshold > 0 && d.material_margin < threshold;
}

static inline float box_iou(const Detection &a, const Detection &b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0.0f;
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0.0f;
}

static inline int decode_output(const int8_t *out, std::vector<Detection> &dets, float output_scale, int output_zero_point) {
    dets.clear();
    if (!out || !std::isfinite(output_scale) || output_scale <= 0 || output_zero_point < -128 || output_zero_point > 127) return -1;
    for (int a = 0; a < 2100; a++) {
        int best = -1;
        float bs = -INFINITY, second = -INFINITY;
        for (int c = 0; c < NCLASS; c++) {
            float s = ((float)out[(4 + c) * 2100 + a] - output_zero_point) * output_scale;
            if (s > bs) { second = bs; bs = s; best = c; }
            else if (s > second) second = s;
        }
        if (bs < CONF_THRES) continue;
        float cx = ((float)out[0 * 2100 + a] - output_zero_point) * output_scale * IMGSZ;
        float cy = ((float)out[1 * 2100 + a] - output_zero_point) * output_scale * IMGSZ;
        float bw = ((float)out[2 * 2100 + a] - output_zero_point) * output_scale * IMGSZ;
        float bh = ((float)out[3 * 2100 + a] - output_zero_point) * output_scale * IMGSZ;
        Detection d;
        d.cls = best;
        d.score = bs;
        d.material_margin = bs - second;
        d.x1 = cx - bw * 0.5f;
        d.y1 = cy - bh * 0.5f;
        d.x2 = cx + bw * 0.5f;
        d.y2 = cy + bh * 0.5f;
        dets.push_back(d);
    }
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b) { return a.score > b.score; });
    std::vector<Detection> keep;
    for (const auto &d : dets) {
        if ((int)keep.size() >= MAX_DET) break;
        bool suppressed = false;
        for (const auto &k : keep)
            if (k.cls == d.cls && box_iou(k, d) > IOU_THRES) { suppressed = true; break; }
        if (!suppressed) keep.push_back(d);
    }
    dets.swap(keep);
    return (int)dets.size();
}

