#include <iostream>
#include <math.h>
#include "common_kernels.cuh"

using namespace std;

int main(int argc, char **argv)
{
    float4 m;
    m.x=1; m.y=2;
    m.z=0; m.w=4;

    float lambda1,lambda2;
    float2 e1,e2;

    compute_eig(m,&lambda1,&lambda2,&e1,&e2);

    cout<<"l1="<<lambda1<<"\tl2="<<lambda2<<endl;
    cout<<"e1x="<<e1.x<<"\te2x="<<e2.x<<endl;
    cout<<"e1y="<<e1.y<<"\te2y="<<e2.y<<endl;

}