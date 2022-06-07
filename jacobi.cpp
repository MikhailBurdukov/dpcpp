#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime> 
#include <cassert>

int main(int argc,char* argv[]){
    class NEOGPUDeviceSelector : public cl::sycl::device_selector {
		public:
		int operator()(const cl::sycl::device &Device) const override {
		using namespace cl::sycl::info;

		const std::string DeviceName = Device.get_info<device::name>();
		const std::string DeviceVendor = Device.get_info<device::vendor>();

      return Device.is_gpu() && (DeviceName.find("MX150") != std::string::npos);
    }
    };
	NEOGPUDeviceSelector Selector;
    assert(argc > 1);
    size_t size = static_cast<size_t>(atoi(argv[1]));
    int cpu_work = static_cast<int>(atoi(argv[2]));
    int gpu_work = 100-cpu_work;
    size_t k = static_cast<size_t>(100);

    std::cout << "Mat size  "<<size<<" cpu "<<cpu_work<<"\n";
    std::vector<double> A(size*size);
    std::vector<double> b(size);
    std::vector<double> x(size,0);
    
    std::fill(A.begin(),A.end(),1);
    std::fill(b.begin(),b.end(),1);
    std::fill(x.begin(),x.end(),0);
    const size_t count  = size*cpu_work/100;
    try {

        sycl::queue q_cpu(sycl::host_selector{});
        sycl::queue q_gpu(Selector);
        auto start = std::chrono::system_clock::now();
        cl::sycl::range<2> Asz{size, size}, Bsz{size,size}, Csz{size, size};
        sycl::buffer<double, 2> d_A(A.data(),sycl::range<2>(size,size));
        sycl::buffer<double, 1> d_X (x.data() ,sycl::range<1>(size)),
                             d_B(b.data(),sycl::range<1>(size));

        for(size_t iter = 0; iter < k; iter++){
            auto prof_cpu = q_cpu.submit([&](sycl::handler& h) {
                sycl::accessor A {d_A, h};
                sycl::accessor b {d_B, h};
                sycl::accessor x {d_X, h};           
                h.parallel_for(sycl::range{count}, [=](sycl::id<1> it) {
                    const int i = it[0];
                    double sum = 0;
                    for (int j = 0; j < size; j++)
                        if(i != j) 
                            sum += A[i][k] * x[i];
                    x[i] = (b[i] - sum) / A[i][i];
                });
            });

            auto prof_gpu = q_cpu.submit([&](sycl::handler& h) {
                sycl::accessor A {d_A, h};
                sycl::accessor b {d_B, h};
                sycl::accessor x {d_X, h};           
                h.parallel_for(sycl::range{size - count}, [=](sycl::id<1> it) {
                    const int i = it[0] + count;
                    double sum = 0;
                    for (int j = 0; j < size; j++)
                        if(i != j) 
                            sum += A[i][k] * x[i];
                    x[i] = (b[i] - sum) / A[i][i];
                });
            });
            q_cpu.wait();
            q_gpu.wait();
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout<< "Okey: time "<<elapsed_seconds.count()<<" s\n"; 
    }
    catch (sycl::exception & e) {
       std::cout << e.what() << std::endl;
       assert(0);
       return 1;
    }
    return 0;
}