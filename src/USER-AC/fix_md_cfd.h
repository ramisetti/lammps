/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(md2cfd,FixMDtoCFD)

#else

#ifndef LMP_FIX_MDTOCFD_H
#define LMP_FIX_MDTOCFD_H

#ifdef ENABLE_MUI
#include "mui.h"
#endif

#include <stdio.h>
#include "fix.h"

namespace LAMMPS_NS {

class Phase{

};

class FixMDtoCFD : public Fix {
 public:
  FixMDtoCFD(class LAMMPS *, int, char **);
  ~FixMDtoCFD();
  int setmask();
  void init();
  void post_constructor();
  void end_of_step();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void update_force(int);
  void update_velocity(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);
  double memory_usage();
  void compute_average_velocity(int ireg, int phase_groupbit);
  void compute_time_averaged_velocity(int ireg);
  void compute_spatial_averaged_velocity(int ireg, int phase_groupbit);
  void C_to_A_coupling(int);
  void C_to_A_coupling_velocity(int);
  virtual void final_integrate();

  
  double *count_one,*count_many,*count_sum;
  double **values_one,**values_many,**values_sum;
  double *count_total,**count_list;
  double **values_total,***values_list;
  char *format;
  

 protected:
  double dtv,dtf,mv_t2f;
  double *step_respa;
  int mass_require;
  int nevery;
  int nfreq,nrepeat,irepeat;

 private:
  double f_xvalue,f_yvalue,f_zvalue;
  double **v_value;
  int varflag,iregion;
  int phase1_groupbit,phase2_groupbit;
  int phase1_region_a_to_c,phase1_region_c_to_a;
  int phase2_region_a_to_c,phase2_region_c_to_a;
  char *xstr,*ystr,*zstr,*estr;
  char *idregion;
  int xvar,yvar,zvar,evar,xstyle,ystyle,zstyle,estyle;
  double foriginal[4],foriginal_all[4];
  double fac_p1_C2A,fac_p1_A2C;
  double fac_p2_C2A,fac_p2_A2C;
  int force_flag;
  int nlevels_respa;

  int maxatom;
  double **sforce;
  double **svelocity;

  int which;
  double t_start,t_stop,t_period,t_target;
  double energy;
  int tstyle,tvar;
  char *tstr;

  char *id_temp;
  class Compute *temperature;
  int tflag;
  int *phase1_A2C_numAtoms, *phase2_A2C_numAtoms;
  int *phase1_C2A_numAtoms, *phase2_C2A_numAtoms;
  int *phase1_nTimeZero, *phase2_nTimeZero;
  double **phase1_A2C_avg_vel_per_proc, **phase1_A2C_global_avg_vel;
  double **phase2_A2C_avg_vel_per_proc, **phase2_A2C_global_avg_vel;

  double **phase1_C2A_avg_vel_per_proc, **phase1_C2A_global_avg_vel;
  double **phase2_C2A_avg_vel_per_proc, **phase2_C2A_global_avg_vel;
  double **phase1_C2A_avg_acc_per_proc, **phase1_C2A_global_avg_acc;
  double **phase2_C2A_avg_acc_per_proc, **phase2_C2A_global_avg_acc;
  int nC2ADataPoints_p1, nA2CDataPoints_p1, nC2ADataPoints_p2, nA2CDataPoints_p2;
  double stream_width;
  double factor_to_OF_vel,factor_to_lammps_vel;
  int startavg;
  int navgTime;
#ifdef ENABLE_MUI
  std::vector<std::string> interfaceNames;
  std::vector<mui::uniface<mui::config_1d>* > interfaces_;  
  int hmm_id;
  int hmm_nIter;  
#endif
  
  FILE *fp;
  bigint nvalid,nvalid_last;
  int nchunk,maxchunk;
  char *idchunk;
  class ComputeChunkAtom *cchunk;  
  class FixAveChunk *fix_ave_chunk;

  void allocate();
  bigint nextvalid();
  
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix hmm does not exist

Self-explanatory.

E: Variable name for fix hmm does not exist

Self-explanatory.

E: Variable for fix hmm is invalid style

Self-explanatory.

E: Cannot use variable energy with constant force in fix hmm

This is because for constant force, LAMMPS can compute the change
in energy directly.

E: Must use variable energy with fix hmm

Must define an energy vartiable when applyting a dynamic
force during minimization.

*/
