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

FixStyle(wenjingimm,FixWenjingIMM)

#else

#ifndef LMP_FIX_WenjingIMM_H
#define LMP_FIX_WenjingIMM_H

#ifdef ENABLE_MUI
#include "mui.h"
#endif

#include "fix.h"

namespace LAMMPS_NS {

class FixWenjingIMM : public Fix {
 public:
  FixWenjingIMM(class LAMMPS *, int, char **);
  ~FixWenjingIMM();
  int setmask();
  void init();
  void end_of_step();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);
  double memory_usage();
  double compute_velocity_x();
  double compute_velocity_y();
  double compute_atom_number();
  virtual void final_integrate();

 protected:
  double dtv,dtf;
  double *step_respa;
  int mass_require;

 private:
  double xvalue,yvalue,zvalue;
  int varflag,iregion,iregion1,iregion2;
  char *xstr,*ystr,*zstr,*estr;
  char *idregion,*idregion1,*idregion2;
  int xvar,yvar,zvar,evar,xstyle,ystyle,zstyle,estyle;
  double foriginal[4],foriginal_all[4];
  int force_flag;
  int nlevels_respa;

  int maxatom;
  double **sforce;

  int which;
  double t_start,t_stop,t_period,t_target;
  double energy;
  int tstyle,tvar;
  char *tstr;

  char *id_temp;
  class Compute *temperature;
  int tflag;
  double velocity_x_per_proc, velocity_y_per_proc, number_per_proc, global_x_velocity, global_y_velocity, global_number;
  double stream_width;
  double factor_to_kg_per_s;
  int startavg;
  int navgTime;
  double eta[3];
  void A_to_C_avg();
  void C_to_A_scaling();
  void C_to_A_coupling();
  double boltz, sigma_lj, epsilon_lj, mass_lj;
  double cfd2md_L, cfd2md_time, cfd2md_temp, cfd2md_v, cfd2md_rho;
  double L_md2cfd, time_md2cfd, temp_md2cfd, v_md2cfd, rho_md2cfd;
  
#ifdef ENABLE_MUI
  std::vector<std::string> interfaceNames;
  std::vector<mui::uniface<mui::config_1d>* > interfaces_;  
  int imm_id;
  int imm_nIter;  
#endif
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix imm does not exist

Self-explanatory.

E: Variable name for fix imm does not exist

Self-explanatory.

E: Variable for fix imm is invalid style

Self-explanatory.

E: Cannot use variable energy with constant force in fix imm

This is because for constant force, LAMMPS can compute the change
in energy directly.

E: Must use variable energy with fix imm

Must define an energy vartiable when applyting a dynamic
force during minimization.

*/
