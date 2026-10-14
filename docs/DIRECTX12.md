# DirectX 9, 11, and 12 backend bring-up

TrSS has separate first-party Windows-native API probes for legacy Direct3D 9, Direct3D 11, and Direct3D 12. Each API has a different graphics/runtime surface; there is no single device call that makes all existing games compatible.

## Current implementation

| API | Native test | What success establishes |
|---|---|---|
| Direct3D 9 | Legacy `IDirect3D9` device creation, back-buffer clear, and system-memory readback | The driver can create this basic D3D9 graphics path and return the test pixel |
| Direct3D 11 | HLSL compute shader, UAV writes, staging-buffer readback | The selected device or WARP can dispatch a simple D3D11 compute shader |
| Direct3D 12 | HLSL compute shader, UAV writes, state transition, queue/fence synchronization, readback | The selected device or WARP can dispatch a simple D3D12 compute shader |

These are API smoke tests. They do **not** establish neural-network inference, game capture, interop between different graphics APIs, compatibility with every game, or Arc B580 performance.

## Build all three probes

From the repository root in a Visual Studio developer environment with CMake and the Windows SDK:

```powershell
cmake -S native/directx12 -B native/directx12/build -A x64
cmake --build native/directx12/build --config Release --parallel
```

The three executables are written to:

- `native/directx12/build/Release/trss_d3d9_probe.exe`
- `native/directx12/build/Release/trss_d3d11_probe.exe`
- `native/directx12/build/Release/trss_dx12_probe.exe`

No prebuilt binary is downloaded or committed.

## Inspect API-specific devices

```powershell
trss graphics-probe --api d3d9 --list-devices
trss graphics-probe --api d3d11 --list-devices
trss graphics-probe --api d3d12 --list-devices
```

If more than one adapter matches a hardware test, use a unique name fragment with `--device-substring`.

## Run hardware tests

```powershell
trss graphics-probe --api d3d9 --self-test --device-substring B580
trss graphics-probe --api d3d11 --self-test --device-substring B580
trss graphics-probe --api d3d12 --self-test --device-substring B580
```

The D3D9 readback test currently requires an X8R8G8B8 or A8R8G8B8 back buffer and fails rather than guessing at unknown pixel formats.

For D3D11 and D3D12, software smoke tests use WARP:

```powershell
trss graphics-probe --api d3d11 --warp-self-test
trss graphics-probe --api d3d12 --warp-self-test
```

WARP validates the API command/resource path but does not measure Intel GPU performance. D3D9 has no corresponding WARP compute test and must use a physical D3D9-capable adapter. The old `dx12-probe` CLI name remains an alias for `graphics-probe`, with D3D12 still the default API.

## What is still required for game support

1. An API-specific method to acquire each supported game's back buffer or swap chain.
2. A safe path to pass color and required auxiliary buffers to the inference engine without expensive or unsafe CPU readbacks.
3. Resource sharing or conversion, synchronization, timestamps, frame pacing, and reset handling for resize, device loss, alt-tab, and multiple-swap-chain applications.
4. Native execution for the actual TrSS model, including weight loading, tensor scheduling, precision, and output composition.
5. Real-game validation on the Arc B580, with image comparisons, latency measurements, and rollback tests.

D3D9 has graphics and render-target functionality but no compute-shader API; D3D11 and D3D12 expose their own compute APIs. A D3D12 compute backend cannot automatically consume a D3D9 or D3D11 texture. The frame-resource path needs a correct shared-resource or copy implementation first.

References: Microsoft's documentation for [Direct3D 9 device creation](https://learn.microsoft.com/en-us/windows/win32/direct3d9/creating-a-device), [Direct3D 11 device creation](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-d3d11createdevice), and [Direct3D 12 device creation](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-d3d12createdevice).
