#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <set>
#include <vector>

namespace one_excess_reference {
using U32=std::uint32_t;using U64=std::uint64_t;
using Row=std::vector<U64>;using Matrix=std::vector<Row>;using Histories=std::vector<Matrix>;using Gate=std::pair<U32,U32>;
inline void check(bool ok,const char* message){if(!ok){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}}
inline Row unit(U32 n,U32 r){Row v((n+63)/64);v[r/64]=U64(1)<<(r%64);return v;}
inline Row sum(const Row& a,const Row& b){Row v(a.size());for(U32 w=0;w<v.size();++w)v[w]=a[w]^b[w];return v;}
struct Numeric {bool operator()(const Row& a,const Row& b)const{return std::lexicographical_compare(a.rbegin(),a.rend(),b.rbegin(),b.rend());}};
struct Expected {std::array<std::int64_t,6> header;std::vector<U32> count;std::vector<Gate> encoder;};
struct Case {Matrix state;Expected expected;U32 resource_phase=0;};
struct BFS {
    U32 n;std::vector<int> distance;std::vector<U64> ways;std::vector<U32> queue;
    explicit BFS(U32 width):n(width),distance(U32(1)<<(n*n),-1),ways(distance.size()){
        U32 identity=0;for(U32 r=0;r<n;++r)identity|=1U<<(r*n+r);distance[identity]=0;ways[identity]=1;queue.push_back(identity);
        for(std::size_t head=0;head<queue.size();++head){U32 m=queue[head];for(U32 c=0;c<n;++c)for(U32 t=0;t<n;++t)if(c!=t){U32 next=m^(((m>>(n*c))&((1U<<n)-1))<<(n*t));
            if(distance[next]<0){distance[next]=distance[m]+1;queue.push_back(next);}if(distance[next]==distance[m]+1){check(~U64(0)-ways[next]>=ways[m],"BFS path count overflow");ways[next]+=ways[m];}}}
        check(queue.size()==(n==2?6:n==3?168:20160),"complete GL BFS");
    }
    Matrix matrix(U32 code)const{Matrix m(n);for(U32 r=0;r<n;++r)m[r]={U64((code>>(r*n))&((1U<<n)-1))};return m;}
};
// Physical-state DP, independent of factors/ordering/lattice/finite differences.
inline U64 orders(const Histories& h){
    U64 size=1;for(const auto& row:h){check(size<=1000000/row.size(),"small physical oracle size");size*=row.size();}
    std::vector<U64> memo(size,~U64(0));std::vector<U32> pos(h.size());
    std::function<U64()> visit=[&](){U64 key=0,stride=1;bool done=true;for(U32 r=0;r<h.size();++r){key+=pos[r]*stride;stride*=h[r].size();done&=pos[r]+1==h[r].size();}
        if(done)return U64(1);if(memo[key]!=~U64(0))return memo[key];U64 total=0;
        for(U32 t=0;t<h.size();++t)if(pos[t]+1<h[t].size()){const Row diff=sum(h[t][pos[t]],h[t][pos[t]+1]);
            for(U32 c=0;c<h.size();++c)if(c!=t&&h[c][pos[c]]==diff){++pos[t];U64 add=visit();--pos[t];check(~U64(0)-total>=add,"physical oracle overflow");total+=add;}}
        return memo[key]=total;
    };return visit();
}
inline Expected reference(const Matrix& m,int distance,U64 ways){
    const U32 n=m.size();std::vector<U32> changed;bool transport=true;for(U32 r=0;r<n;++r){if(m[r]!=unit(n,r))changed.push_back(r);for(U32 c=0;c<n;++c)if(r!=c&&m[r]==unit(n,c))transport=false;}
    if(distance==static_cast<int>(changed.size()))return {{3,static_cast<std::int64_t>(changed.size()),0,0,0,0},{0},{}};
    U64 templates=0,total=0;Histories chosen;
    for(U32 t:changed){std::set<Row,Numeric> endpoints,candidates;for(U32 r=0;r<n;++r)if(r!=t){endpoints.insert(m[r]);endpoints.insert(unit(n,r));}
        for(const auto& e:endpoints){Row mid=sum(m[t],e);if(std::any_of(mid.begin(),mid.end(),[](U64 x){return x!=0;})&&mid!=m[t]&&mid!=unit(n,t))candidates.insert(mid);}
        for(const Row& mid:candidates){if(!endpoints.count(sum(mid,unit(n,t))))continue;++templates;Histories h(n);
            for(U32 r=0;r<n;++r){h[r].push_back(m[r]);if(r==t)h[r].push_back(mid);if(m[r]!=unit(n,r))h[r].push_back(unit(n,r));}
            U64 count=orders(h);check(~U64(0)-total>=count,"template sum overflow");total+=count;if(count&&chosen.empty())chosen=h;}}
    check((total!=0)==(distance==static_cast<int>(changed.size()+1)),"physical histories versus BFS distance");
    if(!total)return {{2,static_cast<std::int64_t>(changed.size()),static_cast<std::int64_t>(templates),-1,0,static_cast<std::int64_t>(changed.size()+1)},{0},{}};
    check(ways==~U64(0)||total==ways,"physical histories versus independent BFS shortest-word count");std::vector<Gate> decoder;
    while(true){bool pending=false,accepted=false;for(const auto& row:chosen)pending|=row.size()>1;if(!pending)break;
        for(U32 t=0;t<n&&!accepted;++t)if(chosen[t].size()>1){Row diff=sum(chosen[t][0],chosen[t][1]);
            for(U32 c=0;c<n;++c)if(c!=t&&chosen[c][0]==diff){Histories trial=chosen;trial[t].erase(trial[t].begin());if(orders(trial)){decoder.emplace_back(c,t);chosen=std::move(trial);accepted=true;}break;}}
        check(accepted,"positive physical count without legal next gate");}
    check(decoder.size()==changed.size()+1,"physical recovery length");std::reverse(decoder.begin(),decoder.end());
    return {{1,static_cast<std::int64_t>(changed.size()),static_cast<std::int64_t>(templates),-1,transport,static_cast<std::int64_t>(decoder.size())},{static_cast<U32>(total),static_cast<U32>(total>>32)},decoder};
}
inline Matrix embed(const Matrix& small,U32 n,const std::vector<U32>& wires){
    Matrix m(n);for(U32 r=0;r<n;++r)m[r]=unit(n,r);for(U32 r=0;r<small.size();++r){std::fill(m[wires[r]].begin(),m[wires[r]].end(),0);
        for(U32 c=0;c<small.size();++c)if(small[r][0]>>c&1)m[wires[r]][wires[c]/64]|=U64(1)<<(wires[c]%64);}return m;
}
inline std::vector<U32> factorial(U32 n,U32 divisor){
    std::vector<U32> digits{1};for(U32 f=2;f<=n;++f){U32 carry=0;for(U32& x:digits){U32 v=x*f+carry;x=v%10;carry=v/10;}while(carry){digits.push_back(carry%10);carry/=10;}}
    U32 rem=0;for(std::size_t i=digits.size();i-->0;){U32 v=rem*10+digits[i];digits[i]=v/divisor;rem=v%divisor;}check(!rem,"decimal factorial quotient");
    std::vector<U32> out;while(!digits.empty()){U64 carry=0;for(std::size_t i=digits.size();i-->0;){U64 v=carry*10+digits[i];digits[i]=static_cast<U32>(v>>32);carry=static_cast<U32>(v);}out.push_back(static_cast<U32>(carry));while(!digits.empty()&&!digits.back())digits.pop_back();}return out;
}
inline Case swap_and_shears(U32 n,bool all_shears=true){
    Matrix m(n);for(U32 r=0;r<n;++r)m[r]=unit(n,r);std::swap(m[0],m[1]);std::vector<Gate> decoder{{1,0},{0,1},{1,0}};U32 separate=0;
    if(all_shears)for(U32 c=2;c+1<n;c+=2){m[c+1]=sum(m[c+1],m[c]);decoder.emplace_back(c,c+1);++separate;}
    std::reverse(decoder.begin(),decoder.end());return {m,{{1,separate+2,2,3,0,separate+3},factorial(separate+3,3),decoder}};
}
inline Case resource(Case c,U32 phase=0){c.resource_phase=phase;c.expected.header={4,c.expected.header[1],0,0,0,0};c.expected.count={0};c.expected.encoder.clear();return c;}
}
