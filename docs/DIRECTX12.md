# Direct3D 12 backend bring-up

TrSS now has a first-party Windows-native Direct3D 12 compute bring-up target. This is a separate backend path, not a translation of the vendored Vulkan project.

## What is implemented

- Enumerates DXGI adapters and reports device/vendor IDs, software-adapter status, and whether a D3D12 feature-level 12_0 device can be created.
- Selects one hardware adapter whose name contains `Arc`, unless a more specific `--device-substring` is supplied. It refuses ambiguous device matches and does not substitute WARP during a hardware test.
- Queries the available shader model and wave-operations capability.
- Compiles a small HLSL compute shader, creates a D3D12 compute root signature and pipeline state, writes to a GPU unordered-access buffer, submits compute work, waits on a fence, reads the output back, and verifies all 1,024 results.
- Runs on Microsoft's WARP software adapter for Windows CI, allowing the Direct3D 12 command/resource/fence path to be exercised without assuming the hosted runner has an Intel GPU.

## Build

From the repository root in a Visual Studio developer environment with CMake and the Windows SDK:

```powershell
cmake -S native/directx12 -B native/directx12/build -A x64
cmake --build native/directx12/build --config Release --parallel
```

The executable is written to `native/directx12/build/Release/trss_dx12_probe.exe`. No prebuilt binary is downloaded or committed.

## Use

List adapters and basic D3D12 capability:

```powershell
trss dx12-probe --list-devices
```

Run the hardware compute roundtrip on the unique Arc adapter:

```powershell
trss dx12-probe --self-test
```

If adapter names are ambiguous or the driver uses a different name, inspect the list and provide a unique substring:

```powershell
trss dx12-probe --self-test --device-substring "B580"
```

Run the software validation path (CI only, not a performance test):

```powershell
trss dx12-probe --warp-self-test
```

When executing the native program directly, its equivalent options are `--list-devices`, `--self-test`, and `--warp-self-test`. An optional `--executable` argument is available on the Python command if the build was placed outside the default path; `TrSS_DX12_PROBE` can set that path as an environment variable.

## What this does not prove

The compute self-test does not load a TrSS checkpoint, execute neural inference, render an image, capture a live game's swapchain, or establish speed/temperature/VRAM behavior on the Arc B580. The native target is a D3D12 execution foundation only.

The existing first-party PyTorch model uses convolution layers, GELU, a context branch, pixel shuffle, and a temporal residual/gate output. A true Direct3D 12 inference backend still needs to implement those operators and their resource/precision contracts, load the matching weights, and produce image comparisons against the same CPU/PyTorch references. Matrix or shader acceleration capability queries are not proof that this network is implemented on that hardware.

The Direct3D 12 API structures and command path follow Microsoft's public documentation for [root signatures](https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures) and [hardware feature levels](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-feature-levels).
