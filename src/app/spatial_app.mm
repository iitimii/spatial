#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "spatial/backend/metal_backend.hpp"
#include "spatial/core/profiler.hpp"
#include "spatial/models/waypoint.hpp"

using namespace spatial;

// Map macOS virtual keyCodes to standard Windows VK / Owl-Control button IDs
static int mac_keycode_to_vk(unsigned short kc) {
    switch (kc) {
        case 13: return 87;   // W
        case 0:  return 65;   // A
        case 1:  return 83;   // S
        case 2:  return 68;   // D
        case 12: return 81;   // Q
        case 14: return 69;   // E
        case 15: return 82;   // R
        case 3:  return 70;   // F
        case 49: return 32;   // Space
        case 56: return 160;  // Left Shift
        case 60: return 161;  // Right Shift
        case 59: return 162;  // Left Ctrl
        case 58: return 164;  // Left Alt
        case 48: return 9;    // Tab
        case 18: return 49;   // 1
        case 19: return 50;   // 2
        case 20: return 51;   // 3
        case 21: return 52;   // 4
        default: return -1;
    }
}

static std::vector<uint8_t> load_ppm_or_synth(const std::string& path, uint32_t w, uint32_t h) {
    if (!path.empty()) {
        std::ifstream ifs(path, std::ios::binary);
        if (ifs) {
            std::string magic;
            uint32_t pw = 0, ph = 0;
            int maxv = 0;
            ifs >> magic >> pw >> ph >> maxv;
            if (magic == "P6" && maxv == 255 && pw == w && ph == h) {
                ifs.get();
                std::vector<uint8_t> out(static_cast<size_t>(w) * h * 3);
                ifs.read(reinterpret_cast<char*>(out.data()), out.size());
                return out;
            }
        }
    }
    // Fallback procedural perspective road scene if no seed PPM given
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            uint8_t r, g, b;
            if (y < h / 2) {
                float t = float(y) / float(h / 2);
                r = uint8_t(45 + 125 * t);
                g = uint8_t(95 + 110 * t);
                b = uint8_t(190 + 55 * t);
            } else {
                float t = float(y - h / 2) / float(h / 2);
                float cx = float(x) - float(w) * 0.5f;
                float road_half = 20.0f + t * 115.0f;
                if (std::fabs(cx) < road_half) {
                    bool stripe = std::fabs(cx) < (3.0f + t * 5.0f);
                    r = stripe ? 230 : 88;
                    g = stripe ? 205 : 86;
                    b = stripe ? 95  : 84;
                } else {
                    r = uint8_t(56 + 28 * t);
                    g = uint8_t(112 + 38 * t);
                    b = uint8_t(48 + 18 * t);
                }
            }
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            rgb[idx + 0] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
    return rgb;
}

static const char* kPresentShader = R"(
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 pos [[position]];
    float2 uv;
};

vertex VOut vs_quad(uint vid [[vertex_id]]) {
    float2 pos[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
    float2 uv[4]  = { float2(0, 1),   float2(1, 1),  float2(0, 0),  float2(1, 0) };
    VOut o;
    o.pos = float4(pos[vid], 0.0, 1.0);
    o.uv  = uv[vid];
    return o;
}

fragment float4 fs_quad(VOut in [[stage_in]],
                        texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    return tex.sample(s, in.uv);
}
)";

struct SharedAppState {
    std::mutex mu;
    std::set<uint8_t> pressed_buttons;
    float accum_dx = 0.0f;
    float accum_dy = 0.0f;
    float accum_scroll = 0.0f;
    bool  reset_requested = false;
    std::atomic<uint32_t> denoise_steps{4};
    std::atomic<bool>     running{true};

    // Double-buffered 4-frame BGRA ring ready for display
    std::mutex frame_mu;
    std::vector<uint8_t> current_batch[4];
    double batch_dt_sec = 0.25;
    double last_dit_ms  = 0.0;
    double last_vae_ms  = 0.0;
    double last_fps     = 0.0;
    uint64_t chunk_id   = 0;
    bool has_new_batch  = false;
};

@interface SpatialView : NSView {
@public
    SharedAppState* appState;
    CAMetalLayer*   metalLayer;
    bool            dragging;
}
@end

@implementation SpatialView

+ (Class)layerClass { return [CAMetalLayer class]; }
- (BOOL)acceptsFirstResponder { return YES; }

- (void)keyDown:(NSEvent*)event {
    unsigned short kc = event.keyCode;
    if (kc == 53) {  // ESC
        appState->running = false;
        [NSApp terminate:nil];
        return;
    }
    if (kc == 15) {  // R = reset world to seed
        std::lock_guard<std::mutex> lk(appState->mu);
        appState->reset_requested = true;
        return;
    }
    if (kc == 19) { appState->denoise_steps = 2; return; }  // '2' = 2-step turbo
    if (kc == 20) { appState->denoise_steps = 3; return; }  // '3' = 3-step balanced
    if (kc == 21) { appState->denoise_steps = 4; return; }  // '4' = 4-step quality

    int vk = mac_keycode_to_vk(kc);
    if (vk >= 0) {
        std::lock_guard<std::mutex> lk(appState->mu);
        appState->pressed_buttons.insert(static_cast<uint8_t>(vk));
    }
}

- (void)keyUp:(NSEvent*)event {
    int vk = mac_keycode_to_vk(event.keyCode);
    if (vk >= 0) {
        std::lock_guard<std::mutex> lk(appState->mu);
        appState->pressed_buttons.erase(static_cast<uint8_t>(vk));
    }
}

- (void)mouseDown:(NSEvent*)event {
    dragging = true;
    std::lock_guard<std::mutex> lk(appState->mu);
    appState->pressed_buttons.insert(1);  // LMB
}

- (void)mouseUp:(NSEvent*)event {
    dragging = false;
    std::lock_guard<std::mutex> lk(appState->mu);
    appState->pressed_buttons.erase(1);
}

- (void)rightMouseDown:(NSEvent*)event {
    std::lock_guard<std::mutex> lk(appState->mu);
    appState->pressed_buttons.insert(2);  // RMB
}

- (void)rightMouseUp:(NSEvent*)event {
    std::lock_guard<std::mutex> lk(appState->mu);
    appState->pressed_buttons.erase(2);
}

- (void)mouseDragged:(NSEvent*)event {
    std::lock_guard<std::mutex> lk(appState->mu);
    appState->accum_dx += static_cast<float>(event.deltaX) * 0.015f;
    appState->accum_dy += static_cast<float>(event.deltaY) * 0.015f;
}

- (void)mouseMoved:(NSEvent*)event {
    if (!dragging) return;
    std::lock_guard<std::mutex> lk(appState->mu);
    appState->accum_dx += static_cast<float>(event.deltaX) * 0.015f;
    appState->accum_dy += static_cast<float>(event.deltaY) * 0.015f;
}

- (void)scrollWheel:(NSEvent*)event {
    std::lock_guard<std::mutex> lk(appState->mu);
    appState->accum_scroll += static_cast<float>(event.scrollingDeltaY);
}

@end

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("Usage: %s <waypoint.spat> <taehv.spat> [seed.ppm] [steps=4]\n", argv[0]);
        std::printf("Controls: WASD=Move | Drag Mouse=Look | Space=Jump | Shift=Sprint | 2/3/4=Denoise Steps | R=Reset | ESC=Quit\n");
        return 1;
    }
    std::string dit_path  = argv[1];
    std::string vae_path  = argv[2];
    std::string seed_path = (argc >= 4) ? argv[3] : "";
    uint32_t init_steps   = (argc >= 5) ? static_cast<uint32_t>(std::stoi(argv[4])) : 4;

    @autoreleasepool {
        MetalBackend backend;
        WorldSession session(backend, dit_path, vae_path, 42);
        const uint32_t W = session.width();
        const uint32_t H = session.height();
        std::vector<uint8_t> seed_rgb = load_ppm_or_synth(seed_path, W, H);

        SharedAppState state;
        state.denoise_steps = init_steps;

        // Pre-allocate 4 unified BGRA GPU buffers for zero-copy TAEHV output
        std::array<Buffer, 4> bgra_bufs;
        for (int i = 0; i < 4; ++i) {
            bgra_bufs[i] = backend.alloc_buffer(static_cast<size_t>(W) * H * 4);
            state.current_batch[i].resize(static_cast<size_t>(W) * H * 4);
        }

        // Seed initial world frame & show seed immediately
        session.seed_from_rgb(seed_rgb.data());
        for (uint32_t i = 0; i < W * H; ++i) {
            uint8_t r = seed_rgb[i * 3 + 0], g = seed_rgb[i * 3 + 1], b = seed_rgb[i * 3 + 2];
            for (int f = 0; f < 4; ++f) {
                state.current_batch[f][i * 4 + 0] = b;
                state.current_batch[f][i * 4 + 1] = g;
                state.current_batch[f][i * 4 + 2] = r;
                state.current_batch[f][i * 4 + 3] = 255;
            }
        }
        state.has_new_batch = true;

        // Launch continuous inference worker thread
        std::thread worker([&]() {
            while (state.running.load()) {
                ControlState ctrl;
                bool do_reset = false;
                {
                    std::lock_guard<std::mutex> lk(state.mu);
                    do_reset = state.reset_requested;
                    state.reset_requested = false;
                    ctrl.mouse_dx = std::clamp(state.accum_dx, -1.0f, 1.0f);
                    ctrl.mouse_dy = std::clamp(state.accum_dy, -1.0f, 1.0f);
                    ctrl.scroll   = state.accum_scroll;
                    state.accum_dx = state.accum_dy = state.accum_scroll = 0.0f;
                    for (uint8_t b : state.pressed_buttons) ctrl.buttons.push_back(b);
                }

                if (do_reset) {
                    session.seed_from_rgb(seed_rgb.data());
                    continue;
                }

                Timer t;
                session.step_bgra(ctrl, bgra_bufs, state.denoise_steps.load());
                double dt_ms = t.elapsed_ms();

                {
                    std::lock_guard<std::mutex> lk(state.frame_mu);
                    for (int f = 0; f < 4; ++f) {
                        std::memcpy(state.current_batch[f].data(), bgra_bufs[f].cpu_ptr,
                                    static_cast<size_t>(W) * H * 4);
                    }
                    state.batch_dt_sec = std::max(dt_ms * 0.001, 0.05);
                    state.last_dit_ms  = session.last_dit_ms();
                    state.last_vae_ms  = session.last_vae_ms();
                    state.last_fps     = 4000.0 / std::max(dt_ms, 1.0);
                    state.chunk_id     = session.chunk_index();
                    state.has_new_batch = true;
                }
            }
        });

        // Native Cocoa window + CAMetalLayer setup
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect frame = NSMakeRect(120, 120, 1024, 512);
        NSWindow* win = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
                        backing:NSBackingStoreBuffered
                          defer:NO];
        [win setTitle:@"Spatial Engine — Waypoint-1.5 (Apple M1 Metal)"];

        SpatialView* view = [[SpatialView alloc] initWithFrame:frame];
        view->appState = &state;
        view->dragging = false;
        [view setWantsLayer:YES];

        id<MTLDevice> dev = (__bridge id<MTLDevice>)backend.raw_device();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.device = dev;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = YES;
        layer.drawableSize = CGSizeMake(1024, 512);
        [view setLayer:layer];
        view->metalLayer = layer;

        [win setContentView:view];
        [win makeKeyAndOrderFront:nil];
        [win makeFirstResponder:view];
        [NSApp activateIgnoringOtherApps:YES];

        // Compile tiny fullscreen quad present pipeline
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kPresentShader]
                                               options:nil
                                                 error:&err];
        MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
        rpd.vertexFunction   = [lib newFunctionWithName:@"vs_quad"];
        rpd.fragmentFunction = [lib newFunctionWithName:@"fs_quad"];
        rpd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        id<MTLRenderPipelineState> rpso = [dev newRenderPipelineStateWithDescriptor:rpd error:&err];

        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                      width:W
                                                                                     height:H
                                                                                  mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> frameTex = [dev newTextureWithDescriptor:td];

        std::vector<uint8_t> display_batch[4];
        for (int i = 0; i < 4; ++i) display_batch[i].resize(static_cast<size_t>(W) * H * 4);
        int sub_frame = 0;
        double sub_dt_sec = 0.066;
        auto last_swap = std::chrono::steady_clock::now();

        // Main event + 60Hz frame pacing loop
        while (state.running.load()) {
            @autoreleasepool {
                NSEvent* ev;
                while ((ev = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                untilDate:[NSDate distantPast]
                                                   inMode:NSDefaultRunLoopMode
                                                  dequeue:YES])) {
                    [NSApp sendEvent:ev];
                }
                if (![win isVisible]) {
                    state.running = false;
                    break;
                }

                bool updated = false;
                double fps = 0.0, dit_ms = 0.0, vae_ms = 0.0;
                uint64_t chunk = 0;
                {
                    std::lock_guard<std::mutex> lk(state.frame_mu);
                    if (state.has_new_batch) {
                        for (int f = 0; f < 4; ++f) display_batch[f] = state.current_batch[f];
                        sub_dt_sec = state.batch_dt_sec / 4.0;
                        sub_frame = 0;
                        state.has_new_batch = false;
                        updated = true;
                        fps = state.last_fps;
                        dit_ms = state.last_dit_ms;
                        vae_ms = state.last_vae_ms;
                        chunk = state.chunk_id;
                    }
                }
                if (updated) {
                    auto mem = get_memory_stats();
                    char title[256];
                    std::snprintf(title, sizeof(title),
                                  "Spatial Engine [M1 Metal] | %.1f FPS | DiT=%.0fms VAE=%.0fms | %u-step | Chunk %llu | %.0f MB | WASD+Drag",
                                  fps, dit_ms, vae_ms, state.denoise_steps.load(),
                                  static_cast<unsigned long long>(chunk), mem.footprint_mb);
                    [win setTitle:[NSString stringWithUTF8String:title]];
                }

                auto now = std::chrono::steady_clock::now();
                double since = std::chrono::duration<double>(now - last_swap).count();
                if (since >= sub_dt_sec && sub_frame < 3) {
                    sub_frame++;
                    last_swap = now;
                }

                [frameTex replaceRegion:MTLRegionMake2D(0, 0, W, H)
                            mipmapLevel:0
                              withBytes:display_batch[sub_frame].data()
                            bytesPerRow:W * 4];

                id<CAMetalDrawable> drawable = [layer nextDrawable];
                if (drawable) {
                    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    pass.colorAttachments[0].texture = drawable.texture;
                    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
                    [enc setRenderPipelineState:rpso];
                    [enc setFragmentTexture:frameTex atIndex:0];
                    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
                    [enc endEncoding];
                    [cb presentDrawable:drawable];
                    [cb commit];
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
        }

        state.running = false;
        if (worker.joinable()) worker.join();
    }
    return 0;
}
