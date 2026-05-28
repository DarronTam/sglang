#include <cstring>
#include <cmath>
#include <iostream>

#include <zenl/zenl.h>

int main() {
  int version = 0;
  zertRet_t status = zenlGetVersion(&version);
  if (status != zertSuccess) {
    std::cerr << zenlGetStatusName(status) << ": "
              << zenlGetErrorString(status) << "\n";
    return 1;
  }

  zenlBuildInfo_t build_info{};
  status = zenlGetBuildInfo(&build_info);
  if (status != zertSuccess) {
    std::cerr << zenlGetStatusName(status) << ": "
              << zenlGetErrorString(status) << "\n";
    return 1;
  }

  zenlRuntimeInfo_t runtime_info{};
  status = zenlGetRuntimeInfo(&runtime_info);
  if (status != zertSuccess) {
    std::cerr << zenlGetStatusName(status) << ": "
              << zenlGetErrorString(status) << "\n";
    return 1;
  }

  zenlHandle_t handle = nullptr;
  status = zenlCreate(&handle);
  if (status != zertSuccess) {
    std::cerr << zenlGetStatusName(status) << ": "
              << zenlGetErrorString(status) << "\n";
    return 1;
  }

  zertStream_t stream = nullptr;
  status = zertStreamCreate(&stream);
  if (status != zertSuccess) {
    zenlDestroy(handle);
    std::cerr << zenlGetStatusName(status) << ": "
              << zenlGetErrorString(status) << "\n";
    return 1;
  }

  status = zenlSetStream(handle, stream);
  if (status != zertSuccess) {
    zertStreamDestroy(stream);
    zenlDestroy(handle);
    std::cerr << zenlGetStatusName(status) << ": "
              << zenlGetErrorString(status) << "\n";
    return 1;
  }

  status = zenlSetAlgoMode(handle, ZENL_ALGO_MODE_DETERMINISTIC);
  if (status != zertSuccess) {
    zertStreamDestroy(stream);
    zenlDestroy(handle);
    std::cerr << zenlGetStatusName(status) << ": "
              << zenlGetErrorString(status) << "\n";
    return 1;
  }

  const unsigned char src_host[] = {1, 2, 3, 4, 5, 6, 7, 8};
  unsigned char dst_host[sizeof(src_host)] = {};
  const float add_a_host[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float add_b_host[] = {10.0f, 20.0f, 30.0f, 40.0f};
  float add_out_host[4] = {};
  size_t memcpy_workspace_size = 1;
  size_t add_workspace_size = 1;
  zenlMemcpyParams_t memcpy_params{};
  memcpy_params.struct_size = sizeof(zenlMemcpyParams_t);
  memcpy_params.nbytes = sizeof(src_host);
  zenlElementwiseParams_t add_params{};
  add_params.struct_size = sizeof(zenlElementwiseParams_t);
  add_params.numel = 4;
  add_params.dtype = ZENL_DTYPE_FLOAT32;
  status = zenlGetWorkspaceSize(handle, ZENL_OP_MEMCPY, &memcpy_params,
                                sizeof(memcpy_params),
                                &memcpy_workspace_size);
  if (status == zertSuccess) {
    status = zenlGetWorkspaceSize(handle, ZENL_OP_ADD, &add_params,
                                  sizeof(add_params), &add_workspace_size);
  }
  void* src_dev = nullptr;
  void* dst_dev = nullptr;
  void* add_a_dev = nullptr;
  void* add_b_dev = nullptr;
  void* add_out_dev = nullptr;

  status = zertMalloc(&src_dev, sizeof(src_host));
  if (status == zertSuccess) {
    status = zertMalloc(&dst_dev, sizeof(src_host));
  }
  if (status == zertSuccess) {
    status = zertMemcpy(src_dev, src_host, sizeof(src_host),
                        zertMemcpyHostToDevice);
  }
  if (status == zertSuccess) {
    status = zenlMemcpyEx(handle, dst_dev, src_dev, sizeof(src_host));
  }
  if (status == zertSuccess) {
    status = zertMalloc(&add_a_dev, sizeof(add_a_host));
  }
  if (status == zertSuccess) {
    status = zertMalloc(&add_b_dev, sizeof(add_b_host));
  }
  if (status == zertSuccess) {
    status = zertMalloc(&add_out_dev, sizeof(add_out_host));
  }
  if (status == zertSuccess) {
    status = zertMemcpy(add_a_dev, add_a_host, sizeof(add_a_host),
                        zertMemcpyHostToDevice);
  }
  if (status == zertSuccess) {
    status = zertMemcpy(add_b_dev, add_b_host, sizeof(add_b_host),
                        zertMemcpyHostToDevice);
  }
  if (status == zertSuccess) {
    status = zenlAddEx(handle, add_out_dev, add_a_dev, add_b_dev, 4, 1.0f,
                       ZENL_DTYPE_FLOAT32);
  }
  if (status == zertSuccess) {
    status = zertStreamSync(stream);
  }
  if (status == zertSuccess) {
    status = zertMemcpy(dst_host, dst_dev, sizeof(dst_host),
                        zertMemcpyDeviceToHost);
  }
  if (status == zertSuccess) {
    status = zertMemcpy(add_out_host, add_out_dev, sizeof(add_out_host),
                        zertMemcpyDeviceToHost);
  }

  if (src_dev) {
    zertFree(src_dev);
  }
  if (dst_dev) {
    zertFree(dst_dev);
  }
  if (add_a_dev) {
    zertFree(add_a_dev);
  }
  if (add_b_dev) {
    zertFree(add_b_dev);
  }
  if (add_out_dev) {
    zertFree(add_out_dev);
  }
  zertStreamDestroy(stream);
  zenlDestroy(handle);

  if (status != zertSuccess) {
    std::cerr << zenlGetStatusName(status) << ": "
              << zenlGetErrorString(status) << "\n";
    return 1;
  }
  if (std::memcmp(src_host, dst_host, sizeof(src_host)) != 0) {
    std::cerr << "zenlMemcpyEx verification failed\n";
    return 1;
  }
  const float add_expected[] = {11.0f, 22.0f, 33.0f, 44.0f};
  for (int i = 0; i < 4; ++i) {
    if (std::fabs(add_out_host[i] - add_expected[i]) > 1e-5f) {
      std::cerr << "zenlAddEx verification failed\n";
      return 1;
    }
  }

  std::cout << "ZENL version: " << version << "\n";
  std::cout << "ZENL ABI version: " << build_info.zenl_abi_version << "\n";
  std::cout << "ZERT ABI version: " << runtime_info.zert_abi_version << "\n";
  std::cout << "Invalid-args name: "
            << zenlGetStatusName(zertErrorArgsInvalid) << "\n";
  std::cout << "zenlMemcpyEx workspace bytes: "
            << memcpy_workspace_size << "\n";
  std::cout << "zenlAddEx workspace bytes: " << add_workspace_size << "\n";
  std::cout << "zenlMemcpyEx smoke: PASS\n";
  std::cout << "zenlAddEx smoke: PASS\n";
  return 0;
}
