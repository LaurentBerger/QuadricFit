
#include<opencv2/opencv.hpp>
#include "opencv2/core/ocl.hpp"
#include<iostream>
using namespace cv;


using namespace cv;

/*
Detecting step edges in noisy SAR images: a new linear operator  IEEE Transactions on Geoscience and Remote Sensing  (Volume:35 ,  Issue: 1 )  1997
*/


class ParallelGradientPaillouYCols: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &im1;
    double w;
    double a;
    bool verbose;
public:
    ParallelGradientPaillouYCols(Mat& imgSrc, Mat &d,double aa,double ww):
        img(imgSrc),
        im1(d),
        w(ww),
        a(aa),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;

        float                *f2;
        int tailleSequence=(img.rows>img.cols)?img.rows:img.cols;
        double *g1=new double[tailleSequence],*g2=new double[tailleSequence];
        int rows=img.rows,cols=img.cols;

        double				q=w/a;
        double				N=sqrt(2/(a*(1+q*q)));
        double				s=sqrt(2*a);
        double				k=sqrt((1+q*q)/(5+q*q));
        double				b1=-2*exp(-a)*cosh(w);
        double				a1=-b1-exp(-2*a)-1;
        double				b2=exp(-2*a);
        double				d=(1-b1+exp(-2*a))/(2*a*exp(-a)*sinh(w)+w*(1-exp(-2*a)));
        double				c1=a*d;
        double				c2=w*d;
        double				a0p=c2;
        double				a1p=(c1*sinh(w)-c2*cosh(w))*exp(-a);
        double				a1m=a1p-c2*b1;
        double				a2m=-c2*b2;

        switch(img.depth()){
        case CV_8U :
        {
            for (int j=range.start;j<range.end;j++)
            {
                // Causal vertical  IIR filter
                unsigned char *c1 = (unsigned char*)img.ptr(0)+j;
                f2 = ((float*)im1.ptr(0))+j;
                double border=*c1;
                g1[0] =  *c1 * (1-b1+b1*b1-b2);
                c1+=cols;
                g1[1] = *c1 - b1*g1[0]-b2*border;
                c1+=cols;
                for (int i=2;i<rows;i++,c1+=cols)
                    g1[i] = *c1-b1*g1[i-1]-b2*g1[i-2];
                // Anticausal vertical IIR filter
                c1 = (unsigned char*)img.ptr(rows-1)+j;
                border=*c1;
                g2[rows-1] =*c1 * (1-b1+b1*b1-b2);
                c1-=cols;
                g2[rows-2] =*c1 - b1*g1[0]-b2*border;
                c1-=cols;
                for (int i=rows-3;i>=0;i--,c1-=cols)
                    g2[i]=*c1-b1*g2[i+1]-b2*g2[i+2];
                for (int i=0;i<rows;i++,f2+=cols)
                    *f2 = (float)(a1*(g1[i]-g2[i]));
            }
        }
            break;
        case CV_16S :
        case CV_16U :
        {
            for (int j=range.start;j<range.end;j++)
            {
                // Causal vertical  IIR filter
                unsigned short *c1 = (unsigned short*)img.ptr(0)+j;
                f2 = ((float*)im1.ptr(0))+j;
                double border=*c1;
                g1[0] =  *c1 * (1-b1+b1*b1-b2);
                c1+=cols;
                g1[1] = *c1 - b1*g1[0]-b2*border;
                c1+=cols;
                for (int i=2;i<rows;i++,c1+=cols)
                    g1[i] = *c1-b1*g1[i-1]-b2*g1[i-2];
                // Anticausal vertical IIR filter
                c1 = (unsigned short*)img.ptr(rows-1)+j;
                border=*c1;
                g2[rows-1] =*c1 * (1-b1+b1*b1-b2);
                c1-=cols;
                g2[rows-2] =*c1 - b1*g1[0]-b2*border;
                c1-=cols;
                for (int i=rows-3;i>=0;i--,c1-=cols)
                    g2[i]=*c1-b1*g2[i+1]-b2*g2[i+2];
                for (int i=0;i<rows;i++,f2+=cols)
                    *f2 = (float)(a1*(g1[i]-g2[i]));
            }
        }
            break;
        case CV_32S :
             break;
        case CV_32F :
             break;
        case CV_64F :
             break;
        default :
            delete []g1;
            delete []g2;
            return ;
            }
        delete []g1;
        delete []g2;
    };
    ParallelGradientPaillouYCols& operator=(const ParallelGradientPaillouYCols &) {
         return *this;
    };
};


class ParallelGradientPaillouYRows: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double a;
    double w;
    bool verbose;

public:
    ParallelGradientPaillouYRows(Mat& imgSrc, Mat &d,double aa,double ww):
        img(imgSrc),
        dst(d),
        a(aa),
        w(ww),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
        float *f1,*f2;
        int tailleSequence=(img.rows>img.cols)?img.rows:img.cols;
        double *g1=new double[tailleSequence],*g2=new double[tailleSequence];
        double k,a5,a6,a7,a8;
        double b3,b4;
        int cols=img.cols;

        double				q=w/a;
        double				N=sqrt(2/(a*(1+q*q)));
        double				s=sqrt(2*a);
        double				k=sqrt((1+q*q)/(5+q*q));
        double				b1=-2*exp(-a)*cosh(w);
        double				a1=-b1-exp(-2*a)-1;
        double				b2=exp(-2*a);
        double				d=(1-b1+exp(-2*a))/(2*a*exp(-a)*sinh(w)+w*(1-exp(-2*a)));
        double				c1=a*d;
        double				c2=w*d;
        double				a0p=c2;
        double				a1p=(c1*sinh(w)-c2*cosh(w))*exp(-a);
        double				a1m=a1p-c2*b1;
        double				a2m=-c2*b2;

        for (int i=range.start;i<range.end;i++)
            {
            f2 = ((float*)dst.ptr(i));
            f1 = ((float*)img.ptr(i));
            int j=0;
            g1[j] = (a5 +a6+b3+b4)* *f1 ;
            g1[j] = (a5 +a6)* *f1 ;
            j++;
            f1++;
            g1[j] = a5 * f1[0]+a6*f1[j-1]+(b3+b4) * g1[j-1];
            g1[j] = a5 * f1[0]+a6*f1[j-1]+(b3) * g1[j-1];
            j++;
            f1++;
            for (j=2;j<cols;j++,f1++)
                g1[j] = a5 * f1[0] + a6 * f1[-1]+b3*g1[j-1]+b4*g1[j-2];
            f1 = ((float*)img.ptr(0));
            f1 += i*cols+cols-1;
            j=cols-1;
            g2[j] = (a7+a8+b3+b4)* *f1;
            g2[j] = (a7+a8)* *f1;
            j--;
            f1--;
            g2[j] = (a7+a8) * f1[1]  +(b3+b4) * g2[j+1];
            g2[j] = (a7+a8) * f1[1]  +(b3) * g2[j+1];
            j--;
            f1--;
            for (j=cols-3;j>=0;j--,f1--)
                g2[j] = a7*f1[1]+a8*f1[2]+b3*g2[j+1]+b4*g2[j+2];
            for (j=0;j<cols;j++,f2++)
                *f2 = (float)(g1[j]+g2[j]);
            }
        delete []g1;
        delete []g2;

    };
    ParallelGradientPaillouYRows& operator=(const ParallelGradientPaillouYRows &) {
         return *this;
    };
};


class ParallelGradientPaillouXCols: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alphaMoyenne;
    bool verbose;

public:
    ParallelGradientPaillouXCols(Mat& imgSrc, Mat &d,double alm):
        img(imgSrc),
        dst(d),
        alphaMoyenne(alm),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {

        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
        float                *f1,*f2;
        int rows=img.rows,cols=img.cols;

        int tailleSequence=(rows>cols)?rows:cols;
        double *g1=new double[tailleSequence],*g2=new double[tailleSequence];
        double k,a5,a6,a7,a8=0;
        double b3,b4;

        k=pow(1-exp(-alphaMoyenne),2.0)/(1+2*alphaMoyenne*exp(-alphaMoyenne)-exp(-2*alphaMoyenne));
        a5=k,a6=k*exp(-alphaMoyenne)*(alphaMoyenne-1);
        a7=k*exp(-alphaMoyenne)*(alphaMoyenne+1),a8=-k*exp(-2*alphaMoyenne);
        b3=2*exp(-alphaMoyenne);
        b4=-exp(-2*alphaMoyenne);

        for (int j=range.start;j<range.end;j++)
        {
            f1 = (float*)img.ptr(0);
            f1+=j;
            int i=0;
            g1[i] = (a5 + a6 +b3+b4)* *f1  ;
            g1[i] = (a5 + a6 )* *f1  ;
            i++;
            f1+=cols;
            g1[i] = a5 * *f1 + a6 * f1[-cols] + (b3+b4) * g1[i-1];
            g1[i] = a5 * *f1 + a6 * f1[-cols] + (b3) * g1[i-1];
            i++;
            f1+=cols;
            for (i=2;i<rows;i++,f1+=cols)
                g1[i] = a5 * *f1 + a6 * f1[-cols] +b3*g1[i-1]+b4 *g1[i-2];
            f1 = (float*)img.ptr(0);
            f1 += (rows-1)*cols+j;
            i = rows-1;
            g2[i] =(a7+a8+b3+b4)* *f1;
            g2[i] =(a7+a8)* *f1;
            i--;
            f1-=cols;
            g2[i] = (a7+a8)* f1[cols] +(b3+b4)*g2[i+1];
            g2[i] = (a7+a8)* f1[cols] +(b3)*g2[i+1];
            i--;
            f1-=cols;
            for (i=rows-3;i>=0;i--,f1-=cols)
                g2[i] = a7*f1[cols] +a8* f1[2*cols]+
                        b3*g2[i+1]+b4*g2[i+2];
            for (i=0;i<rows;i++,f2+=cols)
            {
                f2 = ((float*)dst.ptr(i))+(j*img.channels());
                *f2 = (float)(g1[i]+g2[i]);
            }
        }
        delete []g1;
        delete []g2;
    };
    ParallelGradientPaillouXCols& operator=(const ParallelGradientPaillouXCols &) {
         return *this;
    };
};


class ParallelGradientPaillouXRows: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alphaDerive;
    bool verbose;

public:
    ParallelGradientPaillouXRows(Mat& imgSrc, Mat &d,double ald):
        img(imgSrc),
        dst(d),
        alphaDerive(ald),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
        float *f1;
        int rows=img.rows,cols=img.cols;
        int tailleSequence=(rows>cols)?rows:cols;
        double *g1=new double[tailleSequence],*g2=new double[tailleSequence];
        double kp;;
        double a1,a2,a3,a4;
        double b1,b2;

        kp=pow(1-exp(-alphaDerive),2.0)/exp(-alphaDerive);
        a1=0;
        a2=kp*exp(-alphaDerive);
        a3=-kp*exp(-alphaDerive);
        a4=0;
        b1=2*exp(-alphaDerive);
        b2=-exp(-2*alphaDerive);

        switch(img.depth()){
        case CV_8U :
        case CV_8S :
            {
            unsigned char *c1;
            for (int i=range.start;i<range.end;i++)
                {
                f1 = (float*)dst.ptr(i);
                c1 = (unsigned char*)img.ptr(i);
                int j=0;
                g1[j] = (a1 +a2+b1+b2)* *c1 ;
                g1[j] = (a1 +a2)* *c1 ;
                j++;
                c1++;
                g1[j] = a1 * c1[0]+a2*c1[j-1]+(b1+b2) * g1[j-1];
                g1[j] = a1 * c1[0]+a2*c1[j-1]+(b1) * g1[j-1];
                j++;
                c1++;
                for (j=2;j<cols;j++,c1++)
                    g1[j] = a1 * c1[0] + a2 * c1[-1]+b1*g1[j-1]+b2*g1[j-2];
                c1 = (unsigned char*)img.ptr(0);
                c1 += i*cols+cols-1;
                j=cols-1;
                g2[j] = (a3+a4+b1+b2)* *c1;
                g2[j] = (a3+a4)* *c1;
                j--;
                g2[j] = (a3+a4) * c1[1]  +(b1+b2) * g2[j+1];
                g2[j] = (a3+a4) * c1[1]  +(b1) * g2[j+1];
                j--;
                c1--;
                for (j=cols-3;j>=0;j--,c1--)
                    g2[j] = a3*c1[1]+a4*c1[2]+b1*g2[j+1]+b2*g2[j+2];
                for (j=0;j<cols;j++,f1++)
                    *f1 = (float)(g1[j]+g2[j]);
                }
            }
            break;
        case CV_16S :
        case CV_16U :
            {
            unsigned short *c1;
            f1 = ((float*)dst.ptr(0));
            for (int i=range.start;i<range.end;i++)
                {
                c1 = ((unsigned short*)img.ptr(0));
                c1 += i*cols;
                int j=0;
                g1[j] = (a1 +a2+b1+b2)* *c1 ;
                g1[j] = (a1 +a2)* *c1 ;
                j++;
                c1++;
                g1[j] = a1 * c1[0]+a2*c1[j-1]+(b1+b2) * g1[j-1];
                g1[j] = a1 * c1[0]+a2*c1[j-1]+(b1) * g1[j-1];
                j++;
                c1++;
                for (j=2;j<cols;j++,c1++)
                    g1[j] = a1 * c1[0] + a2 * c1[-1]+b1*g1[j-1]+b2*g1[j-2];
                c1 = ((unsigned short*)img.ptr(0));
                c1 += i*cols+cols-1;
                j=cols-1;
                g2[j] = (a3+a4+b1+b2)* *c1;
                g2[j] = (a3+a4)* *c1;
                j--;
                c1--;
                g2[j] = (a3+a4) * c1[1]  +(b1+b2) * g2[j+1];
                g2[j] = (a3+a4) * c1[1]  +(b1) * g2[j+1];
                j--;
                c1--;
                for (j=cols-3;j>=0;j--,c1--)
                    g2[j] = a3*c1[1]+a4*c1[2]+b1*g2[j+1]+b2*g2[j+2];
                for (j=0;j<cols;j++,f1++)
                    *f1 = (float)(g1[j]+g2[j]);
                }
            }
            break;
        default :
            return ;
            }
        delete []g1;
        delete []g2;
    };
    ParallelGradientPaillouXRows& operator=(const ParallelGradientPaillouXRows &) {
         return *this;
    };
};

UMat GradientPaillouY(UMat op, double a,double w)
{
    Mat tmp(op.size(),CV_32FC(op.channels()));
    UMat imDst(op.rows,op.cols,CV_32FC(op.channels()));
    cv::Mat opSrc = op.getMat(cv::ACCESS_RW);
    cv::Mat dst = imDst.getMat(cv::ACCESS_RW);
    std::vector<Mat> planSrc;
    split(opSrc,planSrc);
    std::vector<Mat> planTmp;
    split(tmp,planTmp);
    std::vector<Mat> planDst;
    split(dst,planDst);
    for (int i = 0; i < static_cast<int>(planSrc.size()); i++)
    {
        if (planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous())
        {
            ParallelGradientPaillouYCols x(planSrc[i],planTmp[i],a,w);
            parallel_for_(Range(0,opSrc.cols), x,getNumThreads());
            ParallelGradientPaillouYRows xr(planTmp[i],planDst[i],a,w);
            parallel_for_(Range(0,opSrc.rows), xr,getNumThreads());

        }
        else
            std::cout << "PB";
    }
    merge(planDst,imDst);
    return imDst;
}

UMat GradientPaillouX(UMat op, double alphaDerive,double alphaMean)
{
    Mat tmp(op.size(),CV_32FC(op.channels()));
    UMat imDst(op.rows,op.cols,CV_32FC(op.channels()));
    cv::Mat opSrc = op.getMat(cv::ACCESS_RW);
    cv::Mat dst = imDst.getMat(cv::ACCESS_RW);
    std::vector<Mat> planSrc;
    split(opSrc,planSrc);
    std::vector<Mat> planTmp;
    split(tmp,planTmp);
    std::vector<Mat> planDst;
    split(dst,planDst);
    for (int i = 0; i < static_cast<int>(planSrc.size()); i++)
    {
        if (planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous())
        {
            ParallelGradientPaillouXRows x(planSrc[i],planTmp[i],alphaDerive);
            parallel_for_(Range(0,opSrc.rows), x,getNumThreads());
            ParallelGradientPaillouXCols xr(planTmp[i],planDst[i],alphaMean);
            parallel_for_(Range(0,opSrc.cols), xr,getNumThreads());
        }
        else
            std::cout << "PB";
    }
    merge(planDst,imDst);
    return imDst;
}
