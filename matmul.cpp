#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime> 
#include <cassert>

#define OUT

void set(int * A, size_t size,int val){
    A = (int *)malloc(sizeof(int)*size * size);
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++) *(A + i*size +j) =val;
    }
    for(int i=0;i<size ;i++){
        for(int j =0;j<size;j++){
            std::cout<< *(A + i*size + j)<<' ';
        }
        std::cout<<std::endl;
    }
}


int set_ij_offset(int size, int cpu_work){
    int count = size*size*cpu_work/100;
    return count;
}



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
	
	
    std::cout << "Mat size  "<<size<<" cpu "<<cpu_work<<"\n";
    std::vector<int> A(size*size);
    std::vector<int> B(size*size);
    std::vector<int> C(size*size);
    std::fill(A.begin(),A.end(),1);
    std::fill(B.begin(),B.end(),1);
    std::fill(C.begin(),C.end(),0);

    // init offsets 
    
    const size_t count = set_ij_offset(size, cpu_work);
    std::cout << count<<std::endl;
    try {

        sycl::queue q_cpu(sycl::host_selector{});
        sycl::queue q_gpu(Selector);
        auto start = std::chrono::system_clock::now();
        
        cl::sycl::range<2> Asz{size, size}, Bsz{size,size}, Csz{size, size};
        sycl::buffer<int, 2> d_A(A.data(),sycl::range<2>(size,size)),
                             d_B(B.data(),sycl::range<2>(size,size)),
                             d_C(C.data(),sycl::range<2>(size,size));
        
        auto prof_cpu = q_cpu.submit([&](sycl::handler& h) {
            sycl::accessor A {d_A, h};
            sycl::accessor B {d_B, h};
            sycl::accessor C {d_C, h};
            h.parallel_for(sycl::range{count}, [=](sycl::id<1> it) {
                const int i = it[0]/size;
                const int j = it[0]%size;
                for (int k = 0; k < size; k++)
                    C[i][j] += A[i][k] * B[k][j];
            });
        });

        auto prof_gpu = q_gpu.submit([&](sycl::handler& h) {
            sycl::accessor A {d_A, h};
            sycl::accessor B {d_B, h};
            sycl::accessor C {d_C, h};
            h.parallel_for(sycl::range{size*size-count}, [=](sycl::id<1> it) {
                const int i = (it[0]+count)/size;
                const int j = (it[0]+count)%size;
                for (int k = 0; k < size; k++)
                    C[i][j] += A[i][k] * B[k][j];
            });
        });

        q_cpu.wait();
        q_gpu.wait();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout<< "Okey: time"<<elapsed_seconds.count()<<"s\n"; 
    }
    catch (sycl::exception & e) {
       std::cout << e.what() << std::endl;
       assert(0);
       return 1;
    }
    return 0;
}