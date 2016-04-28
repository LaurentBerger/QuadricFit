#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>
#include <ctype.h>

using namespace cv;
using namespace std;


UMat GradientDericheY(UMat op, double alphaDerive, double alphaMoyenne);
UMat GradientDericheX(UMat op, double alphaDerive, double alphaMoyenne);
UMat GradientPaillouY(UMat op, double alphaDerive, double alphaMoyenne);
UMat GradientpaillouX(UMat op, double alphaDerive, double alphaMoyenne);
void CannyBis(InputArray _src, OutputArray _dst,double low_thresh, double high_thresh,	int aperture_size, bool L2gradient, InputOutputArray _dx, InputOutputArray _dy);
vector<double> AjusteQuadrique(vector<double> xMarche, vector <double> yMarche,vector<double> paramIni);

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

	// initialize original index locations
	vector<size_t> idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}

class LMSolver : public Algorithm
{
public:
	class Callback
	{
	public:
		virtual ~Callback() {}
		virtual bool compute(InputArray param, OutputArray err, OutputArray J) const = 0;
	};

	virtual void setCallback(const Ptr<LMSolver::Callback>& cb) = 0;
	virtual int run(InputOutputArray _param0) const = 0;
};


class LMSolverImpl : public LMSolver
{
public:
	LMSolverImpl() : maxIters(100) { init(); }
	LMSolverImpl(const Ptr<LMSolver::Callback>& _cb, int _maxIters) : cb(_cb), maxIters(_maxIters) { init(); }

	void init()
	{
		epsx = epsf = FLT_EPSILON;
		printInterval = 0;
	}

	int run(InputOutputArray _param0) const
	{
		Mat param0 = _param0.getMat(), x, xd, r, rd, J, A, Ap, v, temp_d, d;
		int ptype = param0.type();

		CV_Assert((param0.cols == 1 || param0.rows == 1) && (ptype == CV_32F || ptype == CV_64F));
		CV_Assert(cb);

		int lx = param0.rows + param0.cols - 1;
		param0.convertTo(x, CV_64F);

		if (x.cols != 1)
			transpose(x, x);

		if (!cb->compute(x, r, J))
			return -1;
		double S = norm(r, NORM_L2SQR);
		int nfJ = 2;

		mulTransposed(J, A, true);
		gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);

		Mat D = A.diag().clone();

		const double Rlo = 0.25, Rhi = 0.75;
		double lambda = 1, lc = 0.75;
		int i, iter = 0;

		if (printInterval != 0)
		{
			printf("************************************************************************************\n");
			printf("\titr\tnfJ\t\tSUM(r^2)\t\tx\t\tdx\t\tl\t\tlc\n");
			printf("************************************************************************************\n");
		}

		for (;; )
		{
			CV_Assert(A.type() == CV_64F && A.rows == lx);
			A.copyTo(Ap);
			for (i = 0; i < lx; i++)
				Ap.at<double>(i, i) += lambda*D.at<double>(i);
			solve(Ap, v, d, DECOMP_EIG);
			subtract(x, d, xd);
			if (!cb->compute(xd, rd, noArray()))
				return -1;
			nfJ++;
			double Sd = norm(rd, NORM_L2SQR);
			gemm(A, d, -1, v, 2, temp_d);
			double dS = d.dot(temp_d);
			double R = (S - Sd) / (fabs(dS) > DBL_EPSILON ? dS : 1);

			if (R > Rhi)
			{
				lambda *= 0.5;
				if (lambda < lc)
					lambda = 0;
			}
			else if (R < Rlo)
			{
				// find new nu if R too low
				double t = d.dot(v);
				double nu = (Sd - S) / (fabs(t) > DBL_EPSILON ? t : 1) + 2;
				nu = std::min(std::max(nu, 2.), 10.);
				if (lambda == 0)
				{
					invert(A, Ap, DECOMP_EIG);
					double maxval = DBL_EPSILON;
					for (i = 0; i < lx; i++)
						maxval = std::max(maxval, std::abs(Ap.at<double>(i, i)));
					lambda = lc = 1. / maxval;
					nu *= 0.5;
				}
				lambda *= nu;
			}

			if (Sd < S)
			{
				nfJ++;
				S = Sd;
				std::swap(x, xd);
				if (!cb->compute(x, r, J))
					return -1;
				mulTransposed(J, A, true);
				gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);
			}

			iter++;
			bool proceed = iter < maxIters && norm(d, NORM_INF) >= epsx && norm(r, NORM_INF) >= epsf;

			if (printInterval != 0 && (iter % printInterval == 0 || iter == 1 || !proceed))
			{
				printf("%c%10d %10d %15.4e %16.4e %17.4e %16.4e %17.4e\n",
					(proceed ? ' ' : '*'), iter, nfJ, S, x.at<double>(0), d.at<double>(0), lambda, lc);
			}

			if (!proceed)
				break;
		}

		if (param0.size != x.size)
			transpose(x, x);

		x.convertTo(param0, ptype);
		if (iter == maxIters)
			iter = -iter;

		return iter;
	}

	void setCallback(const Ptr<LMSolver::Callback>& _cb) { cb = _cb; }

	Ptr<LMSolver::Callback> cb;

	double epsx;
	double epsf;
	int maxIters;
	int printInterval;
};

Ptr<LMSolver> createLMSolver(const Ptr<LMSolver::Callback>& cb, int maxIters)
{
	return makePtr<LMSolverImpl>(cb, maxIters);
}


class HomographyRefineCallback : public LMSolver::Callback
{
public:
    HomographyRefineCallback(InputArray _src, InputArray _dst)
    {
        src = _src.getMat();
        dst = _dst.getMat();
    }

    bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const
    {
        int i, count = src.checkVector(2);
        Mat param = _param.getMat();
        _err.create(count*2, 1, CV_64F);
        Mat err = _err.getMat(), J;
        if( _Jac.needed())
        {
            _Jac.create(count*2, param.rows, CV_64F);
            J = _Jac.getMat();
            CV_Assert( J.isContinuous() && J.cols == 8 );
        }

        const Point2f* M = src.ptr<Point2f>();
        const Point2f* m = dst.ptr<Point2f>();
        const double* h = param.ptr<double>();
        double* errptr = err.ptr<double>();
        double* Jptr = J.data ? J.ptr<double>() : 0;

        for( i = 0; i < count; i++ )
        {
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            errptr[i*2] = xi - m[i].x;
            errptr[i*2+1] = yi - m[i].y;

            if( Jptr )
            {
                Jptr[0] = Mx*ww; Jptr[1] = My*ww; Jptr[2] = ww;
                Jptr[3] = Jptr[4] = Jptr[5] = 0.;
                Jptr[6] = -Mx*ww*xi; Jptr[7] = -My*ww*xi;
                Jptr[8] = Jptr[9] = Jptr[10] = 0.;
                Jptr[11] = Mx*ww; Jptr[12] = My*ww; Jptr[13] = ww;
                Jptr[14] = -Mx*ww*yi; Jptr[15] = -My*ww*yi;

                Jptr += 16;
            }
        }

        return true;
    }

    Mat src, dst;
};



class QuadriqueLight : public LMSolver::Callback
{
public:

	int nbParameters;
	Mat src, dst;

public:
	QuadriqueLight(InputArray _src, InputArray _dst)
	{
		src = _src.getMat();
		dst = _dst.getMat();
	}

	bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const
	{
		int i, count = dst.cols;
		Mat param = _param.getMat();
		_err.create(count, 1, CV_64F);
		Mat err = _err.getMat(), J;
		if (_Jac.needed())
		{
			_Jac.create(count, param.rows, CV_64F);
			J = _Jac.getMat();
			CV_Assert(J.isContinuous());
		}

		const double* x = src.ptr<double>();
		const double* m = dst.ptr<double>();
		const double* h = param.ptr<double>();
		double* errptr = err.ptr<double>();
		double* Jptr = J.data ? J.ptr<double>() : 0;
		if (nbParameters ==6)
			for (i = 0; i < count; i++)
			{
				double xx = x[i * 3 + 0], yy = x[i * 3 + 1];
				int indexRegion = int(x[3 * i + 2]);
				double xi = h[0] * xx*xx + h[1] * yy*yy + h[2] * xx*yy + h[3] * xx + h[4] * yy + h[5];

				if (Jptr)
				{
					Jptr[0] = xx*xx;
					Jptr[1] = yy*yy;
					Jptr[2] = xx*yy;
					Jptr[3] = xx;
					Jptr[4] = yy;
					Jptr[5] = 1;
					Jptr += nbParameters;
					errptr[i] = (xi - m[i]);
				}
			}
		else
			for (i = 0; i < count; i++)
			{
				double xx = x[i*3+0], yy = x[i*3+1];
				int indexRegion = int(x[3 * i + 2]);
				double xi= h[0] * xx*xx + h[1] * yy*yy + h[2] * xx*yy + h[3] * xx + h[4] * yy + h[5];
			
				if (Jptr)
				{
					Jptr[0] = xx*xx*h[6 + indexRegion] ;
					Jptr[1] = yy*yy*h[6 + indexRegion] ;
					Jptr[2] = xx*yy*h[6 + indexRegion] ;
					Jptr[3] = xx*h[6 + indexRegion] ;
					Jptr[4] = yy*h[6 + indexRegion] ;
					Jptr[5] = (h[6 + indexRegion] );
					for (int j = 6; j<nbParameters; j++)
						Jptr[j] = 0;
					Jptr[6 + indexRegion] = xi ;
					xi = xi * h[6 + indexRegion];
					Jptr += nbParameters;
					errptr[i] = (xi - m[i]);
				}
			}

		return true;
	}


};


static void DisplayImage(UMat x,string s)
{
	vector<Mat> sx;
	split(x, sx);
	vector<double> minVal(3), maxVal(3);
	for (int i = 0; i < static_cast<int>(sx.size()); i++)
	{
		minMaxLoc(sx[i], &minVal[i], &maxVal[i]);
	}
	maxVal[0] = *max_element(maxVal.begin(), maxVal.end());
	minVal[0] = *min_element(minVal.begin(), minVal.end());
	Mat uc;
	x.convertTo(uc, CV_8U,255/(maxVal[0]-minVal[0]),-255*minVal[0]/(maxVal[0]-minVal[0]));
	imshow(s, uc);
}




void DrawTree(Mat &c, vector<vector<Point> > &contours, vector<Vec4i> &hierarchy, vector<char> &fillCtr,int idx, int level)
{

	int i = idx;
	drawContours(c, contours, i, Scalar(i), CV_FILLED);
	fillCtr[i] = 1;
	while (hierarchy[i][0] != -1)
	{
		int j = hierarchy[i][0];
		if (fillCtr[j] == 0)
		{
			drawContours(c, contours, j, Scalar(j), CV_FILLED);
			fillCtr[j] = 1;
			DrawTree(c,contours,hierarchy,fillCtr,j, level);
		}
		i = hierarchy[i][0];

	}
	if (hierarchy[idx][2] != -1)
		DrawTree(c,contours,hierarchy,fillCtr,hierarchy[idx][2], level + 1);
}

double QuadraticError(vector<double> xMarche, vector<double> yMarche, vector<double> pReel)
{
	double e = 0;
	for (int i = 0; i < yMarche.size(); i++)
	{
		double xx = xMarche[3*i];
		double yy = xMarche[3*i+1];
		double p = xx*xx*pReel[0] + yy*yy*pReel[1] + xx*yy*pReel[2] + xx*pReel[3] + yy*pReel[4] + pReel[5];
		e += (p - yMarche[i])*(p - yMarche[i]);
	}
	return e;
}
#ifdef __TOTO__
int main(int argc, char* argv[])
{
	cv::ocl::setUseOpenCL(false);
	UMat m = UMat::zeros(256, 256, CV_8UC1);
	//imread("c:/lib/opencv/samples/data/pic3.png", CV_LOAD_IMAGE_GRAYSCALE).copyTo(m);
	//imread("c:/lib/opencv/samples/data/aero1.jpg", CV_LOAD_IMAGE_GRAYSCALE).copyTo(m);
	Mat mm = m.getMat(ACCESS_RW);
	double pMax = -1000, pMin = 1000;
	vector<double> pReel = { 00,00,40,10,5,100 };
	for (int i = 20; i<mm.rows - 20; i++)
		for (int j = 20; j < mm.cols - 20; j++)
		{
			double xx = (j - 128) / 256.;
			double yy = (i - 128) / 256.;

			double p = xx*xx*pReel[0] + yy*yy*pReel[1] + xx*yy*pReel[2] + xx*pReel[3] + yy*pReel[4] + pReel[5];
			if (p > pMax)
				pMax = p;
			if (p < pMin)
				pMin = p;
			mm.at<uchar>(i, j) = p;
		}
	cout << pMin << "\t" << pMax << "\n";
	int pasPixel = 1;
	int nbC = mm.cols, nbL = mm.rows;
	double	yg = mm.rows / 2, xg = mm.cols / 2;
	vector<double> xMarche;
	vector<double> yMarche[3];
	vector<Mat> plan;
	multimap<int, int> indMarche;
	split(m, plan);
	int ind = 0;
	for (int k = 0; k<plan.size(); k++)
		for (int i = 0; i < mm.rows; i += pasPixel)
		{
			uchar *datar = plan[k].ptr(i);
			for (int j = 0; j < mm.cols; j += pasPixel, datar += pasPixel)
			{
				if (*datar!=0)
				{
					xMarche.push_back((j - xg) / nbC);
					xMarche.push_back((i - yg) / nbL);
					ind = 0;
					xMarche.push_back(ind);
					yMarche[k].push_back(*datar);
				}
			}
		}
	vector <double> paramIni;
	paramIni.resize(6 + indMarche.size());
	cout << "Real parameters :\n";
	for (int i = 0; i < pReel.size(); i++)
		cout << pReel[i] << "\t";
	cout << endl<<"***************************\n";

	paramIni[0] = 0;
	paramIni[1] = 0;
	paramIni[2] = pReel[2]*0.98;
	paramIni[3] = pReel[3]*0.98;
	paramIni[4] = pReel[4]*0.98;
	paramIni[5] = pReel[5]*0.98;
	cout << "Initial guess (LMSolver) :\n";
	for (int i = 0; i < paramIni.size(); i++)
		cout << paramIni[i] << "\t";
	cout << endl;

	Ptr<QuadriqueLight> ql = makePtr < QuadriqueLight>(xMarche, yMarche[0]);
	ql.dynamicCast<QuadriqueLight>()->nbParameters = static_cast<int>(paramIni.size());

	cv::Ptr<LMSolver> pb_ql = createLMSolver(ql, 100 * paramIni.size());
	pb_ql.dynamicCast<LMSolver>()->run(paramIni);
	cout << "Results with LMSolver:\n";
	cout << "Quadratic error :" << QuadraticError(xMarche, yMarche[0], paramIni) << "\n";
	for (int i = 0; i < paramIni.size(); i++)
		cout << paramIni[i] << "\t";
	cout << endl;
	paramIni[0] = pReel[0] * 0.98;
	paramIni[1] = pReel[1] * 0.98;
	paramIni[2] = pReel[2] * 0.98;
	paramIni[3] = pReel[3] * 0.98;
	paramIni[4] = pReel[4] * 0.98;
	paramIni[5] = pReel[5] * 0.98;
	cout << "Initial guess (for Numerical recipes):\n";
	for (int i = 0; i < paramIni.size(); i++)
		cout << paramIni[i] << "\t";
	cout << endl;

	vector <double> pNrc = AjusteQuadrique(xMarche, yMarche[0], paramIni);
	cout << "Results with NRC \n";
	cout << "Quadratic error :" << QuadraticError(xMarche, yMarche[0], pNrc) << "\n";
	for (int i = 0; i < pNrc.size(); i++)
		cout << pNrc[i] << "\t";
	cout << endl;


	paramIni[0] = 0;
	paramIni[1] = 0;
	paramIni[2] = 0;
	paramIni[3] = 0;
	paramIni[4] = 0;
	paramIni[5] = 0;
	cout << "Second try Initial guess (LMSolver):\n";
	for (int i = 0; i < paramIni.size(); i++)
		cout << paramIni[i] << "\t";
	cout << endl;


	pb_ql.dynamicCast<LMSolver>()->run(paramIni);
	cout << "Results with LMSolver for second try :\n";
	cout << "Quadratic error :" << QuadraticError(xMarche, yMarche[0], paramIni) << "\n";
	for (int i = 0; i < paramIni.size(); i++)
		cout << paramIni[i] << "\t";
	cout << endl;
	paramIni[0] = 0;
	paramIni[1] = 0;
	paramIni[2] = 0;
	paramIni[3] = 0;
	paramIni[4] = 0;
	paramIni[5] = 0;
	cout << "second try Initial guess (for Numerical recipes):\n";
	for (int i = 0; i < paramIni.size(); i++)
		cout << paramIni[i] << "\t";
	cout << endl;

	pNrc = AjusteQuadrique(xMarche, yMarche[0], paramIni);
	cout << "Results with NRC for second try :\n";
	cout << "Quadratic error :" << QuadraticError(xMarche, yMarche[0], pNrc) << "\n";
	for (int i = 0; i < pNrc.size(); i++)
		cout << pNrc[i] << "\t";
	cout << endl;




	return 0;
}


#else
int main(int argc, char* argv[])
{
	cv::ocl::setUseOpenCL(false);
	UMat m=UMat::zeros(256,256,CV_8UC1);
	//imread("c:/lib/opencv/samples/data/pic3.png", CV_LOAD_IMAGE_GRAYSCALE).copyTo(m);
	//imread("f:/lib/opencv/samples/data/aero1.jpg", CV_LOAD_IMAGE_GRAYSCALE).copyTo(m);
	imread("C:/Users/Laurent.PC-LAURENT-VISI/Downloads/14607367432299179.png", CV_LOAD_IMAGE_COLOR).copyTo(m);
	imread("C:/Users/Laurent.PC-LAURENT-VISI/Desktop/n67ut.jpg", CV_LOAD_IMAGE_COLOR).copyTo(m);
	/**
	Etape 0 : Calcul du gradient de l'image im (calibre) et moyennage du résultat. L	
	Détection des contours : opérateur de deriche suivi d'un filtre moyenneur
	*/
	double ad=0.8, am = 0.8;
	double thresh1=15, thresh2=50;
	Mat dst; 
	UMat rx = GradientDericheX(m, ad, am);
	UMat ry = GradientPaillouY(m, ad, am);
    UMat dxMax,dyMax;
    if (rx.channels() > 1)
    {
        vector<UMat> x;
        split(rx,x);
        dxMax = x[0].clone();
        for (int i = 1; i<m.channels();i++)
            max( x[i],dxMax,dxMax);
        vector<UMat> y;
        split(ry,y);
        dyMax = y[0].clone();
        for (int i = 1; i<m.channels();i++)
            max(dyMax, y[i],dyMax);

    }
    else
    {
        dxMax=rx;
        dyMax=ry;
    }
    DisplayImage(dxMax, "Gx");
    DisplayImage(dyMax, "Gy");

	Mat dx, dy;
	dxMax.getMat(ACCESS_READ).convertTo(dx, CV_16S);
	dyMax.getMat(ACCESS_READ).convertTo(dy, CV_16S);
	CannyBis(m, dst, thresh1, thresh2, 3, false, dx, dy);
	Mat elt = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(dst, dst, elt);
    bitwise_not(dst,dst);
    imshow("Original", m);
	imshow("CannyDeriche", dst);
	/**
	ETAPE 1  : Régions connexes de l'image (ensemble des pixels avec un gradient faible)
	- Seuillage de l'image gradient : Seuillage par hysterésis de l'image des modules. L'élément apparaissant le plus souvent dans
	l'histogramme de l'image du module du gradient est posMax. Le seuillage par hysterésis estcalculé à partir de
	ratioSeuilMax*posMax et ratioSeuilMin * posMax
	- Recherche des régions connexes à partir de l'image seuillée.
	*/
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<char> fillCtr;

	findContours(dst, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat cmpConnex = Mat::zeros(m.size(), CV_16U);
	fillCtr.resize(contours.size());
	vector<Moments> mu(contours.size());
	double sMax = 0;
	int idxSMax = 0;
	vector<int> surface;
	for (int i = 0; i <contours.size(); i++)
	{
		if (hierarchy[i][3] == -1 && fillCtr[i] == 0)
		{
			DrawTree(cmpConnex,contours,hierarchy,fillCtr,i, 0);
		}
		mu[i] = moments(contours[i]);
		surface.push_back(static_cast<int>(mu[i].m00));
		if (sMax < mu[i].m00)
		{
			sMax = mu[i].m00;
			idxSMax = i;
		}
	}
	vector<size_t> cle=sort_indexes(surface);
	Mat tmp;
	cmpConnex.convertTo(tmp, CV_8U);
	imshow("Cmp Connex", tmp);
	waitKey();
    surface[cmpConnex.at <ushort> (1, 1)]=0;
    surface[cmpConnex.at <ushort> (m.rows-1, m.cols-1)]=0;
	int surfMinMarche = surface[cle[cle.size()-40]];
	int pasPixel = 5;
	int nbC = dst.cols,nbL=dst.rows;
	double	yg = dst.rows / 2, xg = dst.cols / 2;
	vector<double> xMarche;
	vector<double> yMarche[3];
	vector<Mat> plan;
	multimap<int, int> indMarche;
	split(m, plan);
	int ind = 0;
	for (int k=0;k<plan.size();k++)
	    for (int i = 0; i < dst.rows; i+=pasPixel)
	    {
		    uchar *datar = plan[k].ptr(i);
		    for (int j = 0; j < dst.cols; j += pasPixel, datar += pasPixel)
		    {
			    int idxStep = cmpConnex.at <ushort> (i, j);
			    if (surface[idxStep] >= surfMinMarche)
			    {
				    xMarche.push_back((j - xg) / nbC);
				    xMarche.push_back((i - yg) / nbL);
                    multimap<int,int>::iterator it=indMarche.find(idxStep);
                    int ind0;
                    if (it == indMarche.end())
                    {
                        ind0=ind;
					    indMarche.insert(make_pair(idxStep, ind++));
                    }
				    else 
                        ind0 = it->second;
				    xMarche.push_back(ind0);
				    yMarche[k].push_back(*datar);
			    }
		    }
	    }
	vector <double> paramIni(plan.size());
    if (indMarche.size()==1)
	    paramIni.resize(6 );
	else
        paramIni.resize(6 + indMarche.size());

	paramIni[0] = 0;
	paramIni[1] = 0;
	paramIni[2] = 0;
	paramIni[3] =0;
	paramIni[4] = 0;
	paramIni[5] =1;
	for (int i = 6; i < paramIni.size(); i++)
		paramIni[i] = 1;

/*	Ptr<QuadriqueLight> ql = makePtr < QuadriqueLight>(xMarche, yMarche[0]);
		ql.dynamicCast<QuadriqueLight>()->nbParameters = static_cast<int>(paramIni.size());

	cv::Ptr<LMSolver> pb_ql = createLMSolver(ql, 400* paramIni.size());
		pb_ql.dynamicCast<LMSolver>()->run(paramIni);

		for (int i = 0; i < paramIni.size(); i++)
			cout << paramIni[i] << "\t";
		cout << endl;
		cout << endl;
*/
    for (int idxChannel = 0; idxChannel < plan.size(); idxChannel++)
    {
        vector <double> q=AjusteQuadrique(xMarche, yMarche[idxChannel],paramIni);
	    for (int i = 0; i < q.size(); i++)
		    cout << q[i] << "\t";
	    cout << endl;
        double gainGlobale=0;
	    for (int i = 0; i < dst.rows; i++)
	    {
		    uchar *datar = plan[idxChannel].ptr(i);
		    for (int j = 0; j < dst.cols; j++, datar++)
		    {
			    double xx= (j - xg) / nbC;
			    double yy=(i - yg) / nbL;
                gainGlobale+= q[0] * xx*xx + q[1] * yy*yy + q[2] * xx*yy + q[3] * xx + q[4] * yy + q[5];
		    }
	    }
        gainGlobale /= dst.rows*dst.cols;
	    for (int i = 0; i < dst.rows; i++)
	    {
		    uchar *datar = plan[idxChannel].ptr(i);
		    for (int j = 0; j < dst.cols; j++, datar++)
		    {
			    double xx= (j - xg) / nbC;
			    double yy=(i - yg) / nbL;
                double gain= q[0] * xx*xx + q[1] * yy*yy + q[2] * xx*yy + q[3] * xx + q[4] * yy + q[5];
                int idxStep = cmpConnex.at <ushort> (i, j);
                multimap<int,int>::iterator it=indMarche.find(idxStep);
                if (it == indMarche.end())
                    *datar=saturate_cast<uchar>(*datar/(gain/gainGlobale));
                else
                    *datar=saturate_cast<uchar>(*datar/(gain/gainGlobale));
		    }
	    }

    }
    Mat res;
    merge(plan,res);
    imshow("corrected ", res);
    imwrite("corrected.png",res);
    waitKey();
#ifdef TEST_LK_	
	Mat m0=Mat::zeros(256,256,CV_8UC1);
    Mat m1=Mat::zeros(256,256,CV_8UC1);

    Mat mx0,my0,mx1,my1,mt;

    rectangle(m0, Rect(50,50,100,100),64,-1);
    rectangle(m1, Rect(50,51,100,100),64,-1);
    imshow("m0",m0);
    // Gradient x
    Sobel(m0,mx0,CV_16S,1,0);
    // Gradient y
    Sobel(m0,my0,CV_16S,0,1); 
    // gradient t
    subtract(m1, m0, mt, Mat(),CV_16S);
    Mat modGrad;
    mt.convertTo(modGrad,CV_8UC1,1);

    Mat idx;
    imshow("modGrad",modGrad);

    findNonZero(modGrad,idx);
    Mat A(idx.rows,2,CV_32FC1),b(idx.rows,1,CV_32FC1),v;
    for (int i = 0; i<idx.rows; i++)
    {
        A.at<float>(i, 0) = mx0.at<short>(idx.at<Point>(i));
        A.at<float>(i, 1) = my0.at<short>(idx.at<Point>(i));
        b.at<float>(i, 0) = -mt.at<short>(idx.at<Point>(i));
    }
    waitKey();
    SVD s(A);
    cout << A(Rect(0,0,2,5))<<endl;
    cout << b(Rect(0,0,1,5))<<endl;
    s.backSubst(b,v);
    cout << "Speed component : "<<v;
#endif
    return 0;
}  

#endif
