#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"
#include "potential.h"
// for cuda profiler
#include "cuda_profiler_api.h"

//PowerSphericalPotential
//2  arguments: amp, alpha
double KeplerPotentialEval(double R,double Z, double phi, double t, struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    //Get args
    double amp= *args++;
    double alpha= *args;
    //Calculate Rforce
    if ( alpha == 2. )
        return 0.5 * amp * log ( R*R+Z*Z);
    else
        return - amp * pow(R*R+Z*Z,1.-0.5*alpha) / (alpha - 2.);
}
double PowerSphericalPotentialRforce(double R,double Z, double phi,
				      double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate Rforce
  return - amp * R * pow(R*R+Z*Z,-0.5*alpha);
}
double PowerSphericalPotentialPlanarRforce(double R,double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate Rforce
  return - amp * pow(R,-alpha + 1.);
}
double PowerSphericalPotentialzforce(double R,double Z,double phi,
				     double tt,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate zforce
  return - amp * Z * pow(R*R+Z*Z,-0.5*alpha);
}
double PowerSphericalPotentialPlanarR2deriv(double R,double phi,
					     double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate R2deriv
  return amp * (1. - alpha ) * pow(R,-alpha);
}
