// Hand-derived Stage55 residuals and prepared matrices, no host elimination.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
using Word=std::uint64_t;
using Index=std::uint32_t;
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<typename T>struct Buffer{
    T* data=nullptr;std::size_t size;
    explicit Buffer(std::size_t n):size(n){checked(cudaMalloc(reinterpret_cast<void**>(&data),std::max(n,std::size_t{1})*sizeof(T)));checked(cudaMemset(data,0xa5,std::max(n,std::size_t{1})*sizeof(T)));}
    ~Buffer(){cudaFree(data);}
    void upload(const std::vector<T>& values){require(values.size()==size,"upload extent");if(size)checked(cudaMemcpy(data,values.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
    void equals(const std::vector<T>& expected,const char* message){std::vector<T> actual(size);if(size)checked(cudaMemcpy(actual.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));require(actual==expected,message);}
};
Index words(Index n){return (n+63)/64;}
bool bit(const std::vector<Word>& data,Word row,Index stride,Index col){return (data[row*stride+col/64]>>(col%64))&1;}
void set(std::vector<Word>& data,Word row,Index stride,Index col){data[row*stride+col/64]|=Word{1}<<(col%64);}
std::vector<Word> eye(Index n){std::vector<Word> result(Word{n}*words(n),0);for(Index i=0;i<n;++i)set(result,i,words(n),i);return result;}
struct Fixture{
    Index n,m,r,p,t,q;
    std::vector<Word> u,v,a,d,b,c,statistics;
    std::vector<float> residuals;
};
void run(const Fixture& f){
    const Index batch=f.residuals.size(),nw=words(f.n),w=words(f.q+f.r),rw=words(f.r),qw=words(f.q);
    const Word cells=Word{f.p+f.m}*w;
    Buffer<Word> u(f.u.size()),v(f.v.size()),a0(f.a.size()),d0(f.d.size()),matrix_template(cells),prep_scratch(cells),stats(4);
    u.upload(f.u);v.upload(f.v);a0.upload(f.a);d0.upload(f.d);
    Cnot103BilinearPrepareArgs prep{};
    prep.u=f.u.empty()?nullptr:u.data;prep.v=f.v.empty()?nullptr:v.data;prep.a0=f.a.empty()?nullptr:a0.data;prep.d0=f.d.empty()?nullptr:d0.data;
    prep.matrix_template=matrix_template.data;prep.scratch=prep_scratch.data;prep.statistics=stats.data;
    prep.template_words=cells;prep.scratch_words=cells;prep.m=f.m;prep.r=f.r;prep.p=f.p;prep.t=f.t;prep.q=f.q;
    auto short_prep=prep;--short_prep.template_words;
    require(cnot103_cuda_prepare_bilinear(&short_prep)==2,"short I05 template accepted");
    short_prep=prep;--short_prep.scratch_words;
    require(cnot103_cuda_prepare_bilinear(&short_prep)==2,"short I05 setup scratch accepted");
    require(cnot103_cuda_prepare_bilinear(nullptr)==1,"null I05 setup args");
    require(cnot103_cuda_prepare_bilinear(&prep)==0,"I05 setup failed");checked(cudaDeviceSynchronize());
    std::vector<Word> expected_template(cells,0);
    for(Index row=0;row<f.p+f.m;++row)for(Index col=0;col<f.q+f.r;++col){
        bool value=false;
        if(row<f.p)value=col<f.q?bit(f.b,row,qw,col):bit(f.a,row,rw,col-f.q);
        else if(col<f.q)value=bit(f.d,row-f.p,qw,col);
        if(value)set(expected_template,row,w,col);
    }
    matrix_template.equals(expected_template,"I05 literal prepared template differs");stats.equals(f.statistics,"I05 literal fixed ranks/hidden differ");
    u.equals(f.u,"I05 changed U");v.equals(f.v,"I05 changed V");a0.equals(f.a,"I05 changed A0");d0.equals(f.d,"I05 changed D0");
    std::vector<Index> rows(f.m),cols(f.r);
    for(Index i=0;i<f.m;++i)rows[i]=f.n-1-i;
    for(Index i=0;i<f.r;++i)cols[i]=f.r-1-i;
    require(f.m+f.r<=f.n,"literal embedding overlap");
    std::vector<Word> input(Word{batch}*f.n*nw,0);
    for(Index sample=0;sample<batch;++sample){
        for(Index i=0;i<f.n;++i)set(input,Word{sample}*f.n+i,nw,i);
        for(Index i=0;i<f.m;++i)for(Index j=0;j<f.r;++j)
            if(bit(f.c,Word{sample}*f.m+i,rw,j))set(input,Word{sample}*f.n+rows[i],nw,cols[j]);
    }
    Buffer<Word> states(input.size()),scratch(Word{batch}*cells);Buffer<Index>drows(f.m),dcols(f.r);
    Buffer<float> values(2*batch);Buffer<std::uint8_t> status(batch);
    states.upload(input);drows.upload(rows);dcols.upload(cols);
    Cnot103BilinearArgs args{};
    args.states=states.data;args.row_wires=drows.data;args.column_wires=dcols.data;args.matrix_template=matrix_template.data;args.statistics=stats.data;
    args.scratch=scratch.data;args.values=values.data;args.status=status.data;args.template_words=cells;args.scratch_words=scratch.size;
    args.batch=batch;args.dimension=f.n;args.words_per_row=nw;args.m=f.m;args.r=f.r;args.p=f.p;args.q=f.q;
    auto short_args=args;--short_args.scratch_words;require(cnot103_cuda_launch_bilinear(&short_args)==2,"short I05 runtime scratch accepted");
    short_args=args;--short_args.template_words;require(cnot103_cuda_launch_bilinear(&short_args)==2,"short I05 runtime template accepted");
    short_args=args;short_args.words_per_row=0;require(cnot103_cuda_launch_bilinear(&short_args)==1,"invalid I05 state shape accepted");
    require(cnot103_cuda_launch_bilinear(&args)==0,"I05 evaluation failed");checked(cudaDeviceSynchronize());
    std::vector<float> expected;
    for(float rank:f.residuals){expected.push_back(rank);expected.push_back(f.statistics[3]);}
    values.equals(expected,"I05 hand-derived residual/hidden differs");status.equals(std::vector<std::uint8_t>(batch,1),"I05 computed statuses differ");
    states.equals(input,"I05 changed states");matrix_template.equals(expected_template,"I05 changed template");
    // Deliberately injected statistics test only the public exact-f32 guard.
    auto boundary=f.statistics;boundary[3]=Word{1}<<24;stats.upload(boundary);
    require(cnot103_cuda_launch_bilinear(&args)==0,"I05 exact-range launch");checked(cudaDeviceSynchronize());
    for(Index i=0;i<batch;++i)expected[2*i+1]=1<<24;
    values.equals(expected,"I05 exact-f32 boundary rejected");status.equals(std::vector<std::uint8_t>(batch,1),"I05 boundary status");
    ++boundary[3];stats.upload(boundary);
    require(cnot103_cuda_launch_bilinear(&args)==0,"I05 range-guard launch");checked(cudaDeviceSynchronize());
    values.equals(std::vector<float>(2*batch,0),"I05 unavailable payload not zero");status.equals(std::vector<std::uint8_t>(batch,4),"I05 float32 range status");
    std::printf("I05 n=%u m=%u r=%u p=%u t=%u q=%u batch=%u: template, ranks, residuals, guards PASS\n",f.n,f.m,f.r,f.p,f.t,f.q,batch);
}
void small(Index mode,Index n){
    Fixture f{};f.n=n;f.m=f.r=2;
    if(mode==0){ // UV rank1, A3=[0,1], D2=[0;1]: residual is C00+1.
        f.p=f.q=2;f.t=3;f.u={1,2};f.v={1,0,2};f.a={1,2};f.d={1,2};f.b={1,0};f.statistics={1,2,2,0};
    }else if(mode==1){ // UV=0, remaining quotient entry is C01.
        f.p=f.q=2;f.t=4;f.u={1,2};f.v={0,0,1,2};f.a={1,0};f.d={0,1};f.b={0,0};f.statistics={0,1,1,0};
    }else if(mode==2){f.p=f.q=f.t=0;f.statistics={0,0,0,0};}
    else if(mode==3){f.p=0;f.q=f.t=2;f.v={1,2};f.d={1,2};f.statistics={0,0,2,0};}
    else {f.q=0;f.p=f.t=2;f.u={1,2};f.a={1,2};f.statistics={0,2,0,0};}
    for(Index mask=0;mask<16;++mask){
        const Index x=mask&3,y=mask>>2;f.c.push_back(x);f.c.push_back(y);
        const Index rank2=!(x|y)?0:(!x||!y||x==y?1:2);
        f.residuals.push_back(mode==0?1-(mask&1):mode==1?(mask>>1)&1:mode==2?rank2:0);
    }
    run(f);
}
void wide(){
    Fixture f{};f.n=144;f.m=f.r=f.p=f.t=f.q=65;
    f.u=f.v=f.a=f.d=f.b=eye(65);f.statistics={65,65,65,0};f.residuals={65,0,1};
    f.c.assign(3*65*words(65),0);
    for(Index sample=1;sample<3;++sample)for(Index i=0;i<65;++i)set(f.c,Word{sample}*65+i,words(65),i);
    set(f.c,2*65,words(65),64);run(f);
}
int main(){try{for(Index mode=0;mode<5;++mode)small(mode,6);small(0,65);small(1,144);wide();checked(cudaDeviceReset());return 0;}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
