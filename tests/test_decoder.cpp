#include "../deployment/luckfox/decoder.h"
#include <cassert>
#include <cstdio>
#include <limits>

static void anchor(std::vector<int8_t> &buffer, int a, float scale, int zp, int cls, float confidence) {
    const float coordinates[] = {.5f, .5f, .25f, .25f};
    for (int c = 0; c < 4; ++c) buffer[c * 2100 + a] = (int8_t)std::lround(coordinates[c] / scale + zp);
    buffer[(4 + cls) * 2100 + a] = (int8_t)std::lround(confidence / scale + zp);
}

int main() {
    // Two quantizers represent the same boxes/scores. D2 constants must not leak into decoding.
    for (int variant = 0; variant < 2; ++variant) {
        const float scale = variant ? .01f : .005f;
        const int zp = variant ? 0 : -128;
        std::vector<int8_t> buffer(8 * 2100, (int8_t)zp);
        std::vector<Detection> detections;
        assert(decode_output(buffer.data(), detections, scale, zp) == 0);
        anchor(buffer, 0, scale, zp, 0, .8f);
        anchor(buffer, 1, scale, zp, 0, .6f); // same-class duplicate suppressed
        anchor(buffer, 2, scale, zp, 2, .7f); // different material remains
        anchor(buffer, 3, scale, zp, 1, .2f); // below confidence cutoff
        assert(decode_output(buffer.data(), detections, scale, zp) == 2);
        assert(detections[0].cls == 0 && detections[1].cls == 2);
        assert(std::abs(detections[0].score - .8f) < 1e-5f);
        assert(std::abs(detections[0].x1 - 120.f) < 1e-4f);
        assert(std::abs(detections[0].x2 - 200.f) < 1e-4f);
        assert(!unknown_material(detections[0], 0));
        // A confident top class can still have a close runner-up.
        buffer[(4 + 3) * 2100] = (int8_t)std::lround(.72f / scale + zp);
        assert(decode_output(buffer.data(), detections, scale, zp) == 2);
        assert(detections[0].cls == 0 && std::abs(detections[0].material_margin - .08f) < 1e-5f);
        assert(unknown_material(detections[0], .1f) && !unknown_material(detections[0], 0));
        // A tied class score is deferred when enabled; disabling keeps the original first class.
        buffer[(4 + 3) * 2100] = buffer[(4 + 0) * 2100];
        assert(decode_output(buffer.data(), detections, scale, zp) == 2);
        assert(detections[0].cls == 0 && detections[0].material_margin == 0);
        assert(unknown_material(detections[0], .01f) && !unknown_material(detections[0], 0));
        assert(decode_output(buffer.data(), detections, 0, zp) == -1 && detections.empty());
        assert(decode_output(buffer.data(), detections, std::numeric_limits<float>::quiet_NaN(), zp) == -1);
    }
    std::puts("decoder quantization and class-wise NMS checks passed");
}
