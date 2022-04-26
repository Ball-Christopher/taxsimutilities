#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]

//' @export
// [[Rcpp::export]]
NumericVector fast_group_sum(NumericVector x,
                      IntegerVector grp) {
  int n = x.size();
  NumericVector res(n);
  
  double y = 0.0;
  int tgrp = -1;
  int ts = 0;
  
  // Go through linearly
  for(int i=0; i<n; i++){
    if (tgrp != grp[i]){
      for (int s=ts; s<=i; s++){
        res(s) = y;
      }
      tgrp = grp[i];
      ts = i;
      y = 0.0;
    }
    y += x[i];
  }
  
  // Clean up the final group
  for (int s=ts; s<n; s++){
    res(s) = y;
  }
  
  return res;
}

//' @export
// [[Rcpp::export]]
NumericVector fast_group_max(NumericVector x,
                             IntegerVector grp) {
  int n = x.size();
  NumericVector res(n);
  
  double y = 0.0;
  int tgrp = -1;
  int ts = 0;
  
  // Go through linearly
  for(int i=0; i<n; i++){
    if (tgrp != grp[i]){
      for (int s=ts; s<=i; s++){
        res(s) = y;
      }
      tgrp = grp[i];
      ts = i;
      y = x[i];
    }
    if (x[i] > y) y = x[i];
  }
  
  // Clean up the final group
  for (int s=ts; s<n; s++){
    res(s) = y;
  }
  
  return res;
}

//' @export
// [[Rcpp::export]]
NumericVector fast_group_min(NumericVector x,
                             IntegerVector grp) {
  int n = x.size();
  NumericVector res(n);
  
  double y = 0.0;
  int tgrp = -1;
  int ts = 0;
  
  // Go through linearly
  for(int i=0; i<n; i++){
    if (tgrp != grp[i]){
      for (int s=ts; s<=i; s++){
        res(s) = y;
      }
      tgrp = grp[i];
      ts = i;
      y = x[i];
    }
    if (x[i] < y) y = x[i];
  }
  
  // Clean up the final group
  for (int s=ts; s<n; s++){
    res(s) = y;
  }
  
  return res;
}