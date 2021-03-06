#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ECHANGE(x,y) {double zerzer;zerzer=x;x=y;y=zerzer;} 

#include <fstream>
#include <strstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>

using namespace std;

#define TREEL double
#define TESTNIVEAU(x) x<valSeuil

/**
Estimation de la d�riv�e par rapport
� chaque param�tres
*/
void Derive(double *x, double q[], double *y, double *dyda, int na)
{
	double xx = x[0], yy = x[1];
	int indexRegion = int(x[2]);

    if (na==6)
	    {
	    double xx=x[0],yy=x[1];
	    int indexRegion=int(x[2]);

	    y[0] = q[0] * xx*xx + q[1] * yy*yy + q[2] * xx*yy + q[3] * xx + q[4] * yy + q[5];


	    dyda[0] = xx*xx;
	    dyda[1] = yy*yy;
	    dyda[2] = xx*yy;
	    dyda[3] = xx;
	    dyda[4] = yy;
	    dyda[5] = 1;
	    } 
    else	
    // Fin de modif
	    {
	    double xx=x[0],yy=x[1];
	    int indexRegion=int(x[2]);

	    y[0] = q[0]*xx*xx+q[1]*yy*yy+q[2]*xx*yy+q[3]*xx+q[4]*yy+q[5];

	    dyda[0] = xx*xx*q[6+indexRegion];
	    dyda[1] = yy*yy*q[6+indexRegion];
	    dyda[2] = xx*yy*q[6+indexRegion];
	    dyda[3] = xx*q[6+indexRegion];
	    dyda[4] = yy*q[6+indexRegion];
	    dyda[5] = q[6+indexRegion];
	    for (int i=6;i<na;i++)
		    dyda[i] = 0;
	    dyda[6+indexRegion] = y[0] * q[6+indexRegion];
	    y[0] = 	y[0] * q[6+indexRegion];	 
	    }

}

void DeriveHomography(double *x, double q[], double *y, double *dyda, int na)
{
	double xx = x[0], yy = x[1];
	int indexRegion = int(x[2]);

	y[0] = q[0] * xx*xx + q[1] * yy*yy + q[2] * xx*yy + q[3] * xx + q[4] * yy + q[5];

	dyda[0] = xx*xx;
	dyda[1] = yy*yy;
	dyda[2] = xx*yy;
	dyda[3] = xx;
	dyda[4] = yy;
	dyda[5] = 1;

}





#define SWAP(a,b) {TREEL temp=(a);(a)=(b);(b)=temp;}

char gaussj(TREEL **a, long n, TREEL **b, long m)
{
	long *indxc, *indxr, *ipiv;
	long i, icol, irow, j, k, l, ll;
	TREEL big, dum, pivinv;

	indxc = new long[n];
	indxr = new long[n];
	ipiv = new long[n];
	for (j = 0; j<n; j++) ipiv[j] = 0;
	for (i = 0; i<n; i++) {
		big = 0.0;
		for (j = 0; j<n; j++)
			if (ipiv[j] != 1)
				for (k = 0; k<n; k++) {
					if (ipiv[k] == 0) {
						if (fabs(a[j][k]) >= big) {
							big = fabs(a[j][k]);
							irow = j;
							icol = k;
						}
					}
					else if (ipiv[k] > 1) return 1;
				}
		++(ipiv[icol]);
		if (irow != icol) {
			for (l = 0; l<n; l++) SWAP(a[irow][l], a[icol][l])
				for (l = 0; l<m; l++) SWAP(b[irow][l], b[icol][l])
		}
		indxr[i] = irow;
		indxc[i] = icol;
		if (a[icol][icol] == 0.0) return 2;
		pivinv = 1.0 / a[icol][icol];
		a[icol][icol] = 1.0;
		for (l = 0; l<n; l++) a[icol][l] *= pivinv;
		for (l = 0; l<m; l++) b[icol][l] *= pivinv;
		for (ll = 0; ll<n; ll++)
			if (ll != icol) {
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				for (l = 0; l<n; l++) a[ll][l] -= a[icol][l] * dum;
				for (l = 0; l<m; l++) b[ll][l] -= b[icol][l] * dum;
			}
	}
	for (l = n - 1; l >= 0; l--) {
		if (indxr[l] != indxc[l])
			for (k = 0; k<n; k++)
				SWAP(a[k][indxr[l]], a[k][indxc[l]]);
	}
	delete ipiv;
	delete indxr;
	delete indxc;
	return 0;
}

void covsrt(TREEL **covar, long ma, long *lista, long mfit)
{
	int i, j;
	TREEL swap;

	for (j = 0; j<ma - 1; j++)
		for (i = j + 1; i<ma; i++) covar[i][j] = 0.0;
	for (i = 0; i<mfit - 1; i++)
		for (j = i + 1; j<mfit; j++) {
			if (lista[j] > lista[i])
				covar[lista[j]][lista[i]] = covar[i][j];
			else
				covar[lista[i]][lista[j]] = covar[i][j];
		}
	swap = covar[0][0];
	for (j = 0; j<ma; j++) {
		covar[0][j] = covar[j][j];
		covar[j][j] = 0.0;
	}
	covar[lista[0]][lista[0]] = swap;
	for (j = 1; j<mfit; j++) covar[lista[j]][lista[j]] = covar[0][j];
	for (j = 1; j<ma; j++)
		for (i = 0; i <= j - 1; i++) covar[i][j] = covar[j][i];
}

void mrqcof(TREEL *x, TREEL *y, TREEL *sig, long ndata, TREEL *a,
	long ma, long *lista, long mfit, TREEL **alpha, TREEL *beta,
	TREEL *chisq, void(*funcs)(TREEL *x, TREEL a[], TREEL *y, TREEL *dyda, int na))
	/* ANSI: void (*funcs)(TREEL *x,TREEL a[],TREEL *y,TREEL **dyda,int na); */
{
	int k, j, i;
	TREEL ymod, wt, sig2i, dy, *dyda;

	dyda = new double[ma ];
	for (j = 0; j<mfit; j++)
	{
		for (k = 0; k <= j; k++)
			alpha[j][k] = 0.0;
		beta[j] = 0.0;
	}
	*chisq = 0.0;
	for (i = 0; i<ndata; i++)
	{
		(*funcs)(&x[3 * i], a, &ymod, dyda, ma);
		sig2i = (sig[i] * sig[i]);
		dy = y[i] - ymod;
		for (j = 0; j<mfit; j++)
		{
			wt = dyda[lista[j]] * sig2i;
			for (k = 0; k <= j; k++)
				alpha[j][k] += wt*dyda[lista[k]];
			beta[j] += dy*wt;
		}
		(*chisq) += (dy*dy)*sig2i;
	}
	for (j = 1; j<mfit; j++)
		for (k = 0; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];
	delete dyda;
}
char mrqmin(TREEL *x, TREEL *y, TREEL *sig, long ndata, TREEL *a,
	long ma, long *lista, long mfit, TREEL **covar, TREEL **alpha,
	TREEL *chisq, void(*funcs)(TREEL *x, TREEL a[], TREEL *y, TREEL *dyda, int na), TREEL *alamda)
{
	int k, kk, j, ihit;
	static TREEL *da, *atry, **oneda, *beta, ochisq;

	if (*alamda < 0.0) {
		oneda = new double*[mfit];
		for (int i = 0; i<mfit; i++)
			oneda[i] = new double[1];
		atry = new double[ma];
		da = new double[ma];
		beta = new double[ma];
		kk = mfit + 1;
		for (j = 0; j<ma; j++) {
			ihit = 0;
			for (k = 0; k<mfit; k++)
				if (lista[k] == j) ihit++;
			if (ihit == 0)
				lista[kk++] = j;
			else if (ihit > 1) return 2;
		}
		if (kk != ma + 1) return 3;
		*alamda = 0.001;
		mrqcof(x, y, sig, ndata, a,  ma, lista, mfit, alpha, beta, chisq, funcs);
		ochisq = (*chisq);
	}
	for (j = 0; j<mfit; j++) {
		for (k = 0; k<mfit; k++) covar[j][k] = alpha[j][k];
		covar[j][j] = alpha[j][j] * (1.0 + (*alamda));
		oneda[j][0] = beta[j];
	}
	int wwww = gaussj(covar, mfit, oneda, 1);
	if (wwww)
		return 4;
	for (j = 0; j<mfit; j++)
		da[j] = oneda[j][0];
	if (*alamda == 0.0) {
		covsrt(covar, ma, lista, mfit);
		delete beta;
		delete da;
		delete atry;
		for (int i = 0; i<mfit; i++)
			delete oneda[i];
		delete oneda[0];
		return 0;
	}
	for (j = 0; j<ma; j++) atry[j] = a[j];
	for (j = 0; j<mfit; j++)
		atry[lista[j]] = a[lista[j]] + da[j];
	mrqcof(x, y, sig, ndata, atry,  ma, lista, mfit, covar, da, chisq, funcs);
	if (*chisq < ochisq) {
		*alamda *= 0.1;
		ochisq = (*chisq);
		for (j = 0; j<mfit; j++) {
			for (k = 0; k<mfit; k++) alpha[j][k] = covar[j][k];
			beta[j] = da[j];
			a[lista[j]] = atry[lista[j]];
		}
	}
	else {
		*alamda *= 10.0;
		*chisq = ochisq;
	}
	return 0;
}



vector<double> AjusteQuadrique(vector<double> xMarche,vector <double> yMarche,vector<double> paramIni)
{
	// Bruit sur les donn�es
	double	*sig = new double[yMarche.size()];
	// D�riv�e de f(x,y) par rapport aux param�tres
	double	*dyda;
	// Niveau rouge,vert et bleu des pixels dans les marches
	double	*yMarcheR, *yMarcheV, *yMarcheB;
	long	itst;
	long 	ma=paramIni.size(), mfit=paramIni.size();
	double	alamda = -1, chisq, oldchisq;
	// Matrice de covariance et de de gain voir Numerical Recipes
	double	**covar=new double*[ma], **alpha= new double*[ma];
	int	i, j;
	for (int i = 0; i < mfit; i++)
	{
		covar[i] = new double[mfit];
		alpha[i] = new double[mfit];
	}
    for (int i = 0; i<yMarche.size();i++)
        sig[i] = 0.01 * fabs(yMarche[i]);
	int nbPixelsFit = yMarche.size();
	vector<double> q;
	q.resize(paramIni.size());
	q = paramIni;
		// Minimisation
	alamda = -1;
	vector<long> lista(paramIni.size());
	for (int i = 0; i<mfit; i++)
		lista[i] = i;
	int statusMin = mrqmin(xMarche.data(), yMarche.data(), sig, nbPixelsFit, q.data(),  ma, lista.data(), mfit, covar, alpha,
		&chisq, &Derive, &alamda);
	itst = 0;
	int nbIteration = 10;
	while (itst<nbIteration)
	{
		oldchisq = chisq;
		mrqmin(xMarche.data(), yMarche.data(), sig, nbPixelsFit, q.data(),  ma, lista.data(), mfit, covar, alpha,
			&chisq, &Derive, &alamda);
		if (chisq>oldchisq)
			itst = 0;
		else if (fabs(oldchisq - chisq)<FLT_EPSILON)
			itst++;
	}
	return q;
}

vector<double> AjusteHomography(vector<double> x,vector <double> y,vector<double> paramIni)
{
	// Bruit sur les donn�es
	double	*sig = new double[y.size()];
	// D�riv�e de f(x,y) par rapport aux param�tres
	double	*dyda;
	// Niveau rouge,vert et bleu des pixels dans les marches
	long	itst;
	long 	ma=6, mfit=6;
	double	alamda = -1, chisq, oldchisq;
	// Matrice de covariance et de de gain voir Numerical Recipes
	double	**covar=new double*[ma], **alpha= new double*[ma];
	int	i, j;
	for (int i = 0; i < mfit; i++)
	{
		covar[i] = new double[mfit];
		alpha[i] = new double[mfit];
	}
	int nbPixelsFit = y.size();
	vector<double> q;
	q.resize(paramIni.size());
	q = paramIni;
		// Minimisation
	alamda = -1;
	vector<long> lista(6);
	for (int i = 0; i<mfit; i++)
		lista[i] = i;
	int statusMin = mrqmin(x.data(), y.data(), sig, nbPixelsFit, q.data(),  ma, lista.data(), mfit, covar, alpha,
		&chisq, &Derive, &alamda);
	itst = 0;
	int nbIteration = 10;
	while (itst<nbIteration)
	{
		oldchisq = chisq;
		mrqmin(x.data(), y.data(), sig, nbPixelsFit, q.data(),  ma, lista.data(), mfit, covar, alpha,
			&chisq, &DeriveHomography, &alamda);
		if (chisq>oldchisq)
			itst = 0;
		else if (fabs(oldchisq - chisq)<FLT_EPSILON)
			itst++;
	}
	return q;
}


