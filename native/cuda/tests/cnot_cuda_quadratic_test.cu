// Independent oracle: evaluate the Boolean transition constraints numerically
// at 0, each unit vector and each pair. Finite differences recover the full
// quadratic ANF without duplicating the CUDA symbolic equation compiler.
#include "cnot103/cnot_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>
using Word=std::uint64_t;
using Index=std::uint32_t;
using Bit=std::uint8_t;
using Gate=std::pair<Index,Index>;
using Term=std::array<Index,3>;
using Equation=std::vector<Term>;
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<typename T>struct Buffer{
    T* data=nullptr;std::size_t size;
    explicit Buffer(std::size_t n):size(n){checked(cudaMalloc(reinterpret_cast<void**>(&data),std::max(n,std::size_t{1})*sizeof(T)));checked(cudaMemset(data,0xa5,std::max(n,std::size_t{1})*sizeof(T)));}
    ~Buffer(){cudaFree(data);}
    void upload(const std::vector<T>& values){require(values.size()==size,"upload extent");if(size)checked(cudaMemcpy(data,values.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
    void equals(const std::vector<T>& expected,const char* message){
        std::vector<T> actual(size);if(size)checked(cudaMemcpy(actual.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));
        if(actual!=expected){for(std::size_t i=0;i<std::min(actual.size(),expected.size());++i)if(actual[i]!=expected[i]){
            std::fprintf(stderr,"mismatch at %zu: %.0f vs %.0f\n",i,double(actual[i]),double(expected[i]));break;}throw std::runtime_error(message);}
    }
};
struct Fixture {
    Index n,h,vars;
    std::vector<Gate> allowed;
    std::vector<Word> states;
    std::vector<std::vector<Equation>> equations;
    Index groups()const{return equations.front().size();}
    Index batch()const{return equations.size();}
};

std::vector<Bit> residuals(const Fixture& f,const std::vector<Bit>& x,Index sample){
    const Index n=f.n,sw=n*n+3*n+1;
    std::vector<Bit> previous(n*n,0),out;for(Index i=0;i<n;++i)previous[i*n+i]=1;
    Bit previous_active=1;
    for(Index slot=0;slot<f.h;++slot){
        const Index base=slot*sw;
        const Bit active=x[base];
        std::vector<Bit> control(n),target(n),readout(n),next(n*n);
        for(Index i=0;i<n;++i){control[i]=x[base+1+i];target[i]=x[base+1+n+i];readout[i]=x[base+1+2*n+i];}
        for(Index i=0;i<n*n;++i)next[i]=x[base+1+3*n+i];
        Bit control_parity=0,target_parity=0;
        for(Index i=0;i<n;++i){control_parity^=control[i];target_parity^=target[i];}
        out.push_back(active^control_parity);out.push_back(active^target_parity);
        for(Index i=0;i<n;++i)for(Index j=i+1;j<n;++j)out.push_back(control[i]&control[j]);
        for(Index i=0;i<n;++i)for(Index j=i+1;j<n;++j)out.push_back(target[i]&target[j]);
        for(Index i=0;i<n;++i)for(Index j=0;j<n;++j)
            if(std::find(f.allowed.begin(),f.allowed.end(),Gate{i,j})==f.allowed.end())out.push_back(control[i]&target[j]);
        if(slot)out.push_back(active&(1^previous_active));
        for(Index j=0;j<n;++j){Bit dot=0;for(Index i=0;i<n;++i)dot^=control[i]&previous[i*n+j];out.push_back(readout[j]^dot);}
        for(Index i=0;i<n;++i)for(Index j=0;j<n;++j)out.push_back(next[i*n+j]^previous[i*n+j]^(target[i]&readout[j]));
        previous=std::move(next);previous_active=active;
    }
    const Word words=(n+63)/64;
    for(Index i=0;i<n;++i)for(Index j=0;j<n;++j)
        out.push_back(previous[i*n+j]^((f.states[(Word{sample}*n+i)*words+j/64]>>(j%64))&1));
    return out;
}
void derive(Fixture& f){
    for(Index sample=0;sample<f.states.size()/(Word{f.n}*((f.n+63)/64));++sample){
        std::vector<Bit> x(f.vars,0);const auto zero=residuals(f,x,sample);
        std::vector<Equation> equations(zero.size());
        for(Index e=0;e<zero.size();++e)if(zero[e])equations[e].push_back({0,0,0});
        std::vector<std::vector<Bit>> units;
        for(Index i=0;i<f.vars;++i){x[i]=1;units.push_back(residuals(f,x,sample));x[i]=0;
            for(Index e=0;e<zero.size();++e)if(units.back()[e]^zero[e])equations[e].push_back({1,i,0});}
        for(Index i=0;i<f.vars;++i)for(Index j=i+1;j<f.vars;++j){
            x[i]=x[j]=1;const auto pair=residuals(f,x,sample);x[i]=x[j]=0;
            for(Index e=0;e<zero.size();++e)if(pair[e]^units[i][e]^units[j][e]^zero[e])equations[e].push_back({2,i,j});
        }
        // Additional dense assignments check the recovered ANF against the
        // numerical transition constraints, independently of GPU outputs.
        for(Index trial=0;trial<13;++trial){for(Index i=0;i<f.vars;++i)x[i]=((i*17+trial*13)^(i>>2)^trial)&1;
            const auto direct=residuals(f,x,sample);
            for(Index e=0;e<equations.size();++e){Bit value=0;for(auto term:equations[e])
                value^=term[0]==0?1:term[0]==1?x[term[1]]:(x[term[1]]&x[term[2]]);
                require(value==direct[e],"finite difference oracle is not quadratic");}
        }
        f.equations.push_back(std::move(equations));
    }
}
Index dense_rank(std::vector<std::vector<Bit>> rows,Index columns){
    Index rank=0;
    for(Index col=0;col<columns&&rank<rows.size();++col){
        Index pivot=rank;while(pivot<rows.size()&&!rows[pivot][col])++pivot;
        if(pivot==rows.size())continue;std::swap(rows[pivot],rows[rank]);
        for(Index row=rank+1;row<rows.size();++row)if(rows[row][col])
            for(Index c=col;c<columns;++c)rows[row][c]^=rows[rank][c];
        ++rank;
    }return rank;
}
std::vector<Bit> variable_wires(Index n,Index index){
    const Index offset=index%(n*n+3*n+1);std::vector<Bit> support(n,0);
    if(!offset)std::fill(support.begin(),support.end(),1);
    else if(offset<=3*n)support[(offset-1)%n]=1;
    else{Index cell=offset-1-3*n;support[cell/n]=support[cell%n]=1;}return support;
}
void run(const Fixture& f,Index cut,Word prefix_pattern){
    const Index n=f.n,g=f.groups(),b=f.batch(),future=f.vars-cut,max_vars=std::max(f.vars+3,Index{1});
    const Word cap=f.h?n+2:1,channels=max_vars+4;
    std::vector<Bit> prefix(cut);for(Index i=0;i<cut;++i)prefix[i]=(prefix_pattern>>(i%64))&1;
    std::vector<Index> gates;for(auto gate:f.allowed){gates.push_back(gate.first);gates.push_back(gate.second);}
    std::vector<Index> expected_terms(Word{b}*g*cap*3,0),expected_counts(Word{b}*g,0),expected_ranks;
    std::vector<float> expected_values(Word{b}*g*channels,0),expected_support(Word{b}*g*n,0);
    for(Index sample=0;sample<b;++sample){
        std::vector<std::vector<Bit>> beta(g,std::vector<Bit>(future,0)),mixed(cut,std::vector<Bit>(Word{g}*future,0));
        for(Index eq=0;eq<g;++eq){
            const auto& terms=f.equations[sample][eq];require(terms.size()<=cap,"oracle term capacity");
            expected_counts[Word{sample}*g+eq]=terms.size();Bit alpha=0;
            for(Index t=0;t<terms.size();++t){auto term=terms[t];
                for(Index k=0;k<3;++k)expected_terms[((Word{sample}*g+eq)*cap+t)*3+k]=term[k];
                if(!term[0]){alpha^=1;continue;}
                for(Index k=1;k<=term[0];++k){auto support=variable_wires(n,term[k]);for(Index wire=0;wire<n;++wire)
                    if(support[wire])expected_support[(Word{sample}*g+eq)*n+wire]=1;}
                if(term[0]==1){if(term[1]<cut)alpha^=prefix[term[1]];}
                else if(term[2]<cut)alpha^=prefix[term[1]]&prefix[term[2]];
                else if(term[1]<cut){const Index column=term[2]-cut;beta[eq][column]^=prefix[term[1]];mixed[term[1]][Word{eq}*future+column]^=1;}
            }
            const Word at=(Word{sample}*g+eq)*channels;expected_values[at]=eq;expected_values[at+1]=alpha;
            for(Index v=0;v<future;++v)expected_values[at+4+v]=beta[eq][v];
        }
        const Index rb=dense_rank(beta,future),rm=dense_rank(mixed,g*future);expected_ranks.push_back(rb);expected_ranks.push_back(rm);
        for(Index eq=0;eq<g;++eq){expected_values[(Word{sample}*g+eq)*channels+2]=rb;expected_values[(Word{sample}*g+eq)*channels+3]=rm;}
    }
    const Word scratch_per=Word{future}*((g+63)/64)+Word{cut}*((Word{g}*future+63)/64);
    Buffer<Word> states(f.states.size()),scratch(std::max(Word{1},Word{b}*scratch_per));
    Buffer<Index> dgates(gates.size()),terms(expected_terms.size()),counts(expected_counts.size()),ranks(2*b);
    Buffer<Bit> dprefix(prefix.size()),valid(Word{b}*g),status(b);
    Buffer<float> values(expected_values.size()),support(expected_support.size());
    states.upload(f.states);dgates.upload(gates);dprefix.upload(prefix);
    Cnot103QuadraticArgs a{};
    a.states=states.data;a.allowed_gates=gates.empty()?nullptr:dgates.data;a.prefix=cut?dprefix.data:nullptr;
    a.terms=terms.data;a.term_counts=counts.data;a.scratch=scratch.data;a.values=values.data;a.valid=valid.data;
    a.incidence=support.data;a.ranks=ranks.data;a.status=status.data;
    a.state_words=states.size;a.term_elements=terms.size;a.group_elements=valid.size;a.scratch_words=scratch.size;
    a.value_elements=values.size;a.incidence_elements=support.size;a.rank_elements=ranks.size;
    a.batch=b;a.dimension=n;a.words_per_row=(n+63)/64;a.horizon=f.h;a.cut=cut;a.allowed_count=f.allowed.size();a.equations=g;a.max_variables=max_vars;
    for(Index field=0;field<7;++field){auto short_buffer=a;
        switch(field){case 0:--short_buffer.state_words;break;case 1:--short_buffer.term_elements;break;
            case 2:--short_buffer.group_elements;break;case 3:if(!scratch_per)continue;--short_buffer.scratch_words;break;
            case 4:--short_buffer.value_elements;break;case 5:--short_buffer.incidence_elements;break;case 6:--short_buffer.rank_elements;break;}
        require(cnot103_cuda_launch_quadratic(&short_buffer)==2,"short I09 buffer accepted");
    }
    for(Index field=0;field<6;++field){auto invalid=a;
        switch(field){case 0:invalid.cut=f.vars+1;break;case 1:invalid.equations=g+1;break;case 2:invalid.words_per_row=0;break;
            case 3:invalid.max_variables=1<<24;break;case 4:invalid.allowed_count=n*(n-1)+1;break;case 5:invalid.horizon=0xffffffffU;break;}
        require(cnot103_cuda_launch_quadratic(&invalid)==1,"invalid I09 metadata accepted");
    }
    require(cnot103_cuda_launch_quadratic(&a)==0,"I09 launch failed");checked(cudaDeviceSynchronize());
    terms.equals(expected_terms,"I09 full canonical ANF differs");counts.equals(expected_counts,"I09 term counts differ");
    values.equals(expected_values,"I09 summary differs");support.equals(expected_support,"I09 physical support differs");
    ranks.equals(expected_ranks,"I09 exact ranks differ");valid.equals(std::vector<Bit>(Word{b}*g,1),"I09 valid groups differ");
    status.equals(std::vector<Bit>(b,1),"I09 statuses differ");states.equals(f.states,"I09 changed states");dgates.equals(gates,"I09 changed gates");dprefix.equals(prefix,"I09 changed prefix");
}
Fixture fixture(Index n,Index h,Index allowed_mask=0xffffffffU){
    Fixture f{};f.n=n;f.h=h;f.vars=h*(n*n+3*n+1);Index at=0;
    for(Index i=0;i<n;++i)for(Index j=0;j<n;++j)if(i!=j){if((allowed_mask>>(at%32))&1)f.allowed.push_back({i,j});++at;}
    const Word words=(n+63)/64;f.states.assign(2*n*words,0);
    for(Index b=0;b<2;++b)for(Index i=0;i<n;++i)f.states[(Word{b}*n+i)*words+i/64]|=Word{1}<<(i%64);
    if(n>1)f.states[(Word{n}+n-1)*words]^=1;
    if(n>65)f.states[(Word{n}+63)*words+1]^=1;
    derive(f);return f;
}
int main(){try{
    Index cases=0;
    for(Index allowed=0;allowed<4;++allowed){const auto f=fixture(2,1,allowed);
        for(Word prefix=0;prefix<8;++prefix){run(f,3,prefix);++cases;}
        for(Index cut: {0U,11U}){run(f,cut,~Word{0});++cases;}
    }
    for(const auto dims:std::vector<std::pair<Index,Index>>{{1,2},{2,2},{2,7},{4,4},{9,1}}){const auto f=fixture(dims.first,dims.second);
        std::vector<Index> cuts{0,f.vars/2,f.vars};if(f.vars>65){cuts.push_back(63);cuts.push_back(64);cuts.push_back(65);}
        for(Index cut:cuts)for(Word prefix:std::vector<Word>{0,0x5555555555555555ULL,~Word{0}}){run(f,cut,prefix);++cases;}
    }
    for(Index n:{2U,65U,144U}){auto f=fixture(n,0,0);run(f,0,0);++cases;}
    std::printf("I09 %u fixture batches: independent ANF, summaries, rank, support, empty equations and guards PASS\n",cases);
    checked(cudaDeviceReset());return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
