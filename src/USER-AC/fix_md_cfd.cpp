/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "fix_md_cfd.h"
#include "atom.h"
#include "group.h"
#include "atom_masks.h"
#include "accelerator_kokkos.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "fix_ave_chunk.h"
#include "compute_chunk_atom.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixMDtoCFD::FixMDtoCFD(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
#ifdef ENABLE_MUI
  std::string appName_ = arg[3];
  std::string ifsName = arg[4];
  ifsName="ifs"+ifsName;
  interfaceNames.push_back(ifsName);
  interfaces_ = mui::create_uniface<mui::config_1d>(appName_, interfaceNames);
  hmm_nIter=0;
#endif
  if (strcmp(update->unit_style,"lj") == 0) {
    // this is valid only for argon
    factor_to_OF_vel=(3.4e-10/2.15e-12);
    factor_to_lammps_vel=1/factor_to_OF_vel;
    mv_t2f=1.0;
  }
  else if (strcmp(update->unit_style,"real") == 0) {
    factor_to_OF_vel=1.0e5;
    factor_to_lammps_vel=1.0e-5;
    //factor_to_lammps_vel=1.0;
    mv_t2f=1.0/4184;
  }
  else if (strcmp(update->unit_style,"si") == 0) {
    factor_to_OF_vel=1;
    factor_to_lammps_vel=1;
    mv_t2f=4184; // to do
  }
  else if (strcmp(update->unit_style,"metal") == 0) {
    factor_to_OF_vel=1.0e2;
    factor_to_lammps_vel=1.0e-2;
    mv_t2f=4184; // to do
  }
  else error->all(FLERR,"no units command!");

  if (narg < 5) error->all(FLERR,"Illegal fix md2cfd command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;

  xstr = ystr = zstr = NULL;

  f_xvalue=0;
  f_yvalue=0;
  f_zvalue=0;
  v_value=NULL;

  xstyle = CONSTANT;
  ystyle = NONE;
  zstyle = NONE;

  // optional args
  nevery = 1;
  iregion = -1;
  idregion = NULL;
  phase1_groupbit = -1;
  phase2_groupbit = -1;
  phase1_region_a_to_c = -1;
  phase1_region_c_to_a = -1;
  phase2_region_a_to_c = -1;
  phase2_region_a_to_c = -1;

  nC2ADataPoints_p1 = 1;
  nA2CDataPoints_p1 = 1;
  nC2ADataPoints_p2 = 1;
  nA2CDataPoints_p2 = 1;
  fac_p1_C2A = 1;
  fac_p1_A2C = 1;

  estr = NULL;
//fix_ave_chunk = NULL;

  startavg=0;
  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix hmm command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix hmm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"startavg") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix hmm command");
      startavg = atoi(arg[iarg+1]);
      if (startavg <= 0) error->all(FLERR,"Illegal fix hmm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix hmm command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix hmm does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"phase1") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix hmm command");
      int phase1_group = group->find(arg[iarg+1]);
      if (phase1_group == -1) error->all(FLERR,"Could not find fix group ID");
      phase1_groupbit = group->bitmask[phase1_group];
      phase1_region_a_to_c = domain->find_region(arg[iarg+2]);
      phase1_region_c_to_a = domain->find_region(arg[iarg+3]);
      nC2ADataPoints_p1 = atoi(arg[iarg+4]);
      nA2CDataPoints_p1 = atoi(arg[iarg+5]);
      std::cout << " Number of data points to transfer from C to A :" << nC2ADataPoints_p1 << " and A to C :" << nA2CDataPoints_p1 << std::endl;
      Region *region = NULL;
      if (phase1_region_c_to_a >= 0) {
        region = domain->regions[phase1_region_c_to_a];
	region->prematch();
      }
      fac_p1_C2A=nC2ADataPoints_p1/(region->extent_yhi-region->extent_ylo);
      fac_p1_A2C=nA2CDataPoints_p1/(region->extent_yhi-region->extent_ylo);
      if (phase1_region_a_to_c == -1 || phase1_region_c_to_a == -1)
        error->all(FLERR,"Phase1 region ID for fix hmm does not exist");
      iarg += 6;
    } else if (strcmp(arg[iarg],"phase2") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix hmm command");
      int phase2_group = group->find(arg[iarg+1]);
      if (phase2_group == -1) error->all(FLERR,"Could not find fix group ID");
      phase2_groupbit = group->bitmask[phase2_group];
      phase2_region_a_to_c = domain->find_region(arg[iarg+2]);
      phase2_region_c_to_a = domain->find_region(arg[iarg+3]);
      nC2ADataPoints_p2 = atoi(arg[iarg+4]);
      nA2CDataPoints_p2 = atoi(arg[iarg+5]);
      Region *region = NULL;
      if (phase1_region_c_to_a >= 0) {
        region = domain->regions[phase1_region_c_to_a];
	region->prematch();
      }
      fac_p2_C2A=nC2ADataPoints_p2/(region->extent_yhi-region->extent_ylo);
      fac_p2_A2C=nA2CDataPoints_p2/(region->extent_yhi-region->extent_ylo);
      if (phase2_region_c_to_a == -1 || phase2_region_a_to_c == -1)
        error->all(FLERR,"Phase2 region IDs for fix hmm does not exist");
      iarg += 6;
    } else if (strcmp(arg[iarg],"energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix hmm command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        int n = strlen(&arg[iarg+1][2]) + 1;
        estr = new char[n];
        strcpy(estr,&arg[iarg+1][2]);
      } else error->all(FLERR,"Illegal fix hmm command");
      iarg += 2;
    } else if(strcmp(arg[iarg],"streamwidth") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix hmm command");
      stream_width = atof(arg[iarg+1]);
      if (stream_width <= 0) error->all(FLERR,"Illegal fix hmm command");
      iarg += 2;
    }
    else error->all(FLERR,"Illegal fix hmm command");
  }

  navgTime=nevery-startavg+1;

  force_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  if(stream_width==0)
	  error->all(FLERR,"fix hmm command needs streamwidth to be defined");

  fp=NULL;
  nfreq = nevery;
  //nrepeat = 0.5*nfreq;
  nrepeat = nevery;

  // create a new compute chunk style
  // id = fix-ID + temp, compute group = fix group
  format = (char *) " %g";
  cchunk = NULL;
  idchunk = new char[10];
  strcpy(idchunk,"_cchunk_");
  char **newarg = new char*[9];
  newarg[0] = idchunk;
  newarg[1] = "all";
  newarg[2] = (char *) "chunk/atom";
  newarg[3] = (char *) "bin/1d";
  newarg[4] = (char *) "y";
  newarg[5] = (char *) "lower";
  newarg[6] = (char *) "3.0";
  newarg[7] = (char *) "units";
  newarg[8] = (char *) "box";
  modify->add_compute(9,newarg);
  delete [] newarg;

  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Could not find compute ID for atom/chunk");
  cchunk = (ComputeChunkAtom *) modify->compute[icompute];

  if (nrepeat > 1) cchunk->lockcount++;

  irepeat = 0;
  count_one = count_many = count_sum = count_total = NULL;
  values_one = values_many = values_sum = values_total = NULL;
  maxchunk = 0;
  nchunk = 1;
  allocate();

  nvalid_last = -1;
  nvalid = nextvalid();
  modify->addstep_compute_all(nvalid);

  maxatom = 1;
  memory->create(sforce,maxatom,4,"hmm:sforce");
  memory->create(svelocity,maxatom,3,"hmm:svelocity");
  memory->create(v_value,nC2ADataPoints_p1,3,"hmm:v_value");
  memory->create(phase1_A2C_avg_vel_per_proc,nA2CDataPoints_p1,3,"hmm:a2c_avg_vel_per_proc_p1");
  memory->create(phase1_A2C_global_avg_vel,nA2CDataPoints_p1,3,"hmm:a2c_global_avg_vel_p1");
  memory->create(phase2_A2C_avg_vel_per_proc,nA2CDataPoints_p1,3,"hmm:a2c_avg_vel_per_proc_p2");
  memory->create(phase2_A2C_global_avg_vel,nA2CDataPoints_p1,3,"hmm:a2c_global_avg_vel_p2");
  memory->create(phase1_C2A_avg_vel_per_proc,nC2ADataPoints_p1,3,"hmm:c2a_avg_vel_per_proc_p1");
  memory->create(phase1_C2A_global_avg_vel,nC2ADataPoints_p1,3,"hmm:c2a_global_avg_vel_p1");
  memory->create(phase2_C2A_avg_vel_per_proc,nC2ADataPoints_p1,3,"hmm:c2a_avg_vel_per_proc_p2");
  memory->create(phase2_C2A_global_avg_vel,nC2ADataPoints_p1,3,"hmm:c2a_global_avg_vel_p2");
  memory->create(phase1_C2A_avg_acc_per_proc,nC2ADataPoints_p1,3,"hmm:c2a_avg_acc_per_proc_p1");
  memory->create(phase1_C2A_global_avg_acc,nC2ADataPoints_p1,3,"hmm:c2a_global_avg_acc_p1");
  memory->create(phase2_C2A_avg_acc_per_proc,nC2ADataPoints_p1,3,"hmm:c2a_avg_acc_per_proc_p2");
  memory->create(phase2_C2A_global_avg_acc,nC2ADataPoints_p1,3,"hmm:c2a_global_avg_acc_p2");

  memory->create(phase1_A2C_numAtoms,nA2CDataPoints_p1,"hmm:a2c_numatoms_p1");
  memory->create(phase2_A2C_numAtoms,nA2CDataPoints_p1,"hmm:a2c_numAtoms_p2");
  memory->create(phase1_C2A_numAtoms,nC2ADataPoints_p1,"hmm:c2a_numAtoms_p1");
  memory->create(phase2_C2A_numAtoms,nC2ADataPoints_p1,"hmm:c2a_numAtoms_p2");
  memory->create(phase1_nTimeZero,nA2CDataPoints_p1,"hmm:nTimeZero_p1");
  memory->create(phase2_nTimeZero,nA2CDataPoints_p1,"hmm:nTimeZero_p2");  
}

/* ---------------------------------------------------------------------- */

FixMDtoCFD::~FixMDtoCFD()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  delete [] estr;
  delete [] idregion;
#ifdef _MUI
  for(size_t i=0; interfaces_.size(); i++)
	  delete interfaces_[i];
#endif
  memory->destroy(sforce);
  memory->destroy(svelocity);
  memory->destroy(v_value);
  memory->destroy(phase1_A2C_avg_vel_per_proc);
  memory->destroy(phase2_A2C_avg_vel_per_proc);
  memory->destroy(phase1_A2C_global_avg_vel);
  memory->destroy(phase2_A2C_global_avg_vel);
  memory->destroy(phase1_C2A_avg_vel_per_proc);
  memory->destroy(phase2_C2A_avg_vel_per_proc);
  memory->destroy(phase1_C2A_global_avg_vel);
  memory->destroy(phase2_C2A_global_avg_vel);
  memory->destroy(phase1_C2A_avg_acc_per_proc);
  memory->destroy(phase2_C2A_avg_acc_per_proc);
  memory->destroy(phase1_C2A_global_avg_acc);
  memory->destroy(phase2_C2A_global_avg_acc);
  memory->destroy(phase1_A2C_numAtoms);
  memory->destroy(phase2_A2C_numAtoms);
  memory->destroy(phase1_C2A_numAtoms);
  memory->destroy(phase2_C2A_numAtoms);
  memory->destroy(phase1_nTimeZero);
  memory->destroy(phase2_nTimeZero);

    
  // decrement lock counter in compute chunk/atom, it if still exists
  if (nrepeat > 1) {
    int icompute = modify->find_compute(idchunk);
    if (icompute >= 0) {
      cchunk = (ComputeChunkAtom *) modify->compute[icompute];      
      cchunk->lockcount--;
    }
  }
  delete [] idchunk;
  idchunk = NULL;
  
  fp = NULL;
  count_one = NULL;
  count_many = NULL;
  count_sum = NULL;
  count_total = NULL;
  values_one = NULL;
  values_many = NULL;
  values_sum = NULL;
  values_total = NULL;
  idchunk = NULL;
  cchunk = NULL;
  
}


void FixMDtoCFD::post_constructor()
{
  /*
  char **newarg = new char*[9];
  int n = strlen(id) + strlen("_AVE_CHUNK") + 1;
  char *id_fix_ = new char[n];
  strcpy(id_fix_,id);
  strcpy(id_fix_,"_AVE_CHUNK");
  newarg[0] = id_fix_;
  newarg[1] = group->names[0];
  newarg[2] = (char *) "ave/chunk";
  newarg[3] = (char *) "1";
  newarg[4] = (char *) "100000";
  newarg[5] = (char *) "200000";
  newarg[6] = (char *) "cc1";
  newarg[7] = (char *) "vx";
  
  modify->add_fix(8,newarg);
  fix_ave_chunk = (FixAveChunk *) modify->fix[modify->nfix-1];
  //restartFlag = modify->fix[modify->nfix-1]->restart_reset;

  delete [] newarg;
  */
}


/* ---------------------------------------------------------------------- */

int FixMDtoCFD::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= FINAL_INTEGRATE;
  mask |= POST_FORCE;
  mask |= END_OF_STEP;
  mask |= THERMO_ENERGY;
  //  mask |= POST_FORCE_RESPA;
  //  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMDtoCFD::init()
{
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix hmm does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
    else error->all(FLERR,"Variable for fix hmm is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix hmm does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
    else error->all(FLERR,"Variable for fix hmm is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix hmm does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
    else error->all(FLERR,"Variable for fix hmm is invalid style");
  }
  if (estr) {
    evar = input->variable->find(estr);
    if (evar < 0)
      error->all(FLERR,"Variable name for fix hmm does not exist");
    if (input->variable->atomstyle(evar)) estyle = ATOM;
    else error->all(FLERR,"Variable for fix hmm is invalid style");
  } else estyle = NONE;

  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix hmm does not exist");
  }

  if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

  if (varflag == CONSTANT && estyle != NONE)
    error->all(FLERR,"Cannot use variable energy with "
               "constant force in fix hmm");
  if ((varflag == EQUAL || varflag == ATOM) &&
      update->whichflag == 2 && estyle == NONE)
    error->all(FLERR,"Must use variable energy with fix hmm");

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

// NVE
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;
  
#ifdef ENABLE_MUI
  for ( size_t i =  0; i <  interfaces_.size(); i++ ) 
  {
    int lc=std::stoi(interfaces_[i]->getIFS().substr(3,1));
    for(int j=0; j<nC2ADataPoints_p1;j++){
      v_value[j][0]=interfaces_[i]->fetch( "phase1_vel_x_", i*nC2ADataPoints_p1+j, hmm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() );
      v_value[j][1]=interfaces_[i]->fetch( "phase1_vel_y_", i*nC2ADataPoints_p1+j, hmm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() ); 
      //v_value[j][1]=interfaces_[i]->fetch( "phase1_vel_y_", i*nC2ADataPoints_p1+j, hmm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() );
      v_value[j][0]*=factor_to_lammps_vel;
      //v_value[j][1]*=factor_to_lammps_vel*0.0;
      v_value[j][1]*=0.0;
      if (comm->me == 0)
	fprintf(stdout,"\nInitial LAMMPS fetch vel (lammps units): %f %f ifs name: %s location: %d iteration: %d timestep (lammps): %d\n", v_value[j][0], v_value[j][1], interfaces_[i]->getIFS().c_str(), lc, hmm_nIter, update->ntimestep);
      
    }
  }
#endif 
}

/* ---------------------------------------------------------------------- */

void FixMDtoCFD::final_integrate()
{  
}

/* ---------------------------------------------------------------------- */

void FixMDtoCFD::end_of_step()
{
  // compute_average_velocity(phase1_region_a_to_c,phase1_groupbit);
  // if(phase2_region_a_to_c>=0)
  //   compute_average_velocity(phase2_region_a_to_c,phase2_groupbit);
  
  compute_time_averaged_velocity(phase1_region_a_to_c);
  
#ifdef ENABLE_MUI
  if (update->ntimestep%nevery==0){
    for ( size_t i =  0; i <  interfaces_.size(); i++ ) 
    {
      int lc=std::stoi(interfaces_[i]->getIFS().substr(3,1));
      mui::point1d loc(lc);
      for(int j=0; j<nA2CDataPoints_p1;j++)
      {
	interfaces_[i]->push( "phase1_md_avg_vel_x_", i*nA2CDataPoints_p1+j, phase1_A2C_global_avg_vel[j][0]);
	interfaces_[i]->push( "phase1_md_avg_vel_y_", i*nA2CDataPoints_p1+j, phase1_A2C_global_avg_vel[j][1]);
	if (comm->me == 0)
	  fprintf(stdout,"\nLAMMPS push vel : %f %f ifs name: %s  location: %d iteration: %d timestep: %d \n", phase1_A2C_global_avg_vel[j][0], phase1_A2C_global_avg_vel[j][1], interfaces_[i]->getIFS().c_str(), lc, hmm_nIter, update->ntimestep);
      }
      interfaces_[i]->commit(hmm_nIter);
    }
   
    if (comm->me == 0){
      std::ofstream myfile;
      myfile.open (interfaces_[0]->getIFS(), std::ios::out | std::ios::app);
      if(update->ntimestep==nevery)
	myfile << "IFS: " << interfaces_[0]->getIFS() << "\n";
      for(int j=0; j<nC2ADataPoints_p1;j++){
	myfile << "C2A: time " << update->ntimestep << " patch index " << j << " velocity (lammps units) x: " << v_value[j][0] << " y: " << v_value[j][1] << "\n";
      }
      for(int j=0; j<nA2CDataPoints_p1;j++){
	myfile << "A2C: time " << update->ntimestep << " global_vel (SI units) x: " << phase1_A2C_global_avg_vel[j][0] << " y: " << phase1_A2C_global_avg_vel[j][1] << "\n";
      }
      myfile.close();
    }
    hmm_nIter++;

    if(update->ntimestep!=update->laststep)
    {
      for ( size_t i =  0; i <  interfaces_.size(); i++ ) 
      {
	int lc=std::stoi(interfaces_[i]->getIFS().substr(3,1));
	for(int j=0; j<nC2ADataPoints_p1;j++)
	{
	  v_value[j][0]=interfaces_[i]->fetch( "phase1_vel_x_", i*nC2ADataPoints_p1+j, hmm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() ); 
	  v_value[j][1]=interfaces_[i]->fetch( "phase1_vel_y_", i*nC2ADataPoints_p1+j, hmm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() );
	  v_value[j][0]*=factor_to_lammps_vel;
	  //v_value[j][1]*=factor_to_lammps_vel;
	  v_value[j][1]*=0.0;
	  if (comm->me == 0)
	    fprintf(stdout,"\nLAMMPS fetch vel (lammps units): %f %f ifs name: %s location: %d iteration: %d timestep: %d\n", v_value[j][0], v_value[j][1], interfaces_[i]->getIFS().c_str(), lc, hmm_nIter,update->ntimestep);
	}
      }
    }
  }
#endif
}

/* ---------------------------------------------------------------------- */

void FixMDtoCFD::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }

}

/* ---------------------------------------------------------------------- */

void FixMDtoCFD::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixMDtoCFD::post_force(int vflag)
{
  // imm requires force updates
  //update_force(vflag);

  //update_velocity(vflag);
  
  //C_to_A_coupling(vflag);
  C_to_A_coupling_velocity(vflag);
}

void FixMDtoCFD::C_to_A_coupling(int vflag) {
  
  compute_spatial_averaged_velocity(phase1_region_c_to_a,phase1_groupbit);

  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  double *mass = atom->mass;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  
  // update region if necessary
  
  Region *region = NULL;
  if (phase1_region_c_to_a >= 0) {
    region = domain->regions[phase1_region_c_to_a];
    region->prematch();
  }

  // reallocate svelocity array if necessary
  
  if (varflag == ATOM && atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(svelocity);
    memory->create(svelocity, maxatom, 3, "setvelocity:svelocity");
  }

  foriginal[0] = foriginal[1] = foriginal[2] = 0.0;
  force_flag = 0;

  double tmp[nC2ADataPoints_p1][3];
  //if(update->ntimestep%nevery==0)
    for (int j=0; j< nC2ADataPoints_p1; j++){
      
      tmp[j][0] = -phase1_C2A_global_avg_acc[j][0] + mv_t2f*mass[type[j]]*(v_value[j][0]-phase1_C2A_global_avg_vel[j][0])/update->dt;
      tmp[j][1] = -phase1_C2A_global_avg_acc[j][1] + mv_t2f*mass[type[j]]*(v_value[j][1]-phase1_C2A_global_avg_vel[j][1])/update->dt;
      tmp[j][2] = -phase1_C2A_global_avg_acc[j][2] + mv_t2f*mass[type[j]]*(v_value[j][2]-phase1_C2A_global_avg_vel[j][2])/update->dt;
      
      //tmp[j][0] = (v_value[j][0]-phase1_C2A_global_avg_vel[j][0]);
      //tmp[j][1] = (v_value[j][1]-phase1_C2A_global_avg_vel[j][1]);

      // tmp[j][0] = v_value[j][0];
      // tmp[j][1] = v_value[j][1];
      
      //fprintf(stdout,"TT %d %d %f %f \n",update->ntimestep,j,tmp[j][0],tmp[j][1]);
    }

  int C2A_ind=0;
  if (varflag == CONSTANT) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & phase1_groupbit) {
	if (region && !region->match(x[i][0], x[i][1], x[i][2]))
	  continue;
	
	// does not work for two phase systems
	C2A_ind = fac_p1_C2A*(x[i][1]-region->extent_ylo);
	
	if(C2A_ind<0 || C2A_ind>=nC2ADataPoints_p1) fprintf(stdout, "BIG ERROR %d %d\n", update->ntimestep, C2A_ind);
	
	//f[i][0] += mass[type[i]]*tmp[C2A_ind][0];
	//f[i][1] += mass[type[i]]*tmp[C2A_ind][1];
	
	foriginal[0] += f[i][0];
	foriginal[1] += f[i][1];
	foriginal[2] += f[i][2];

	if (xstyle) {
	  //v[i][0] = tmp[C2A_ind][0];
	  f[i][0] += tmp[C2A_ind][0];
	  //fprintf(stdout, " TYPE %d %d MASS %f\n", i, type[i], mass[type[i]]);
	}
	if (ystyle) {	  
	  //v[i][1] = tmp[C2A_ind][1];
	  f[i][1] += tmp[C2A_ind][1];
	}
	if (zstyle) {
	  //v[i][2] = tmp[C2A_ind][2];
	  f[i][2] += tmp[C2A_ind][2];
	}
      }
 
    // variable force, wrap with clear/add
  }
  /*  else {
    
    modify->clearstep_compute();
    
    if (xstyle == EQUAL)
      v_value[0][0]= input->variable->compute_equal(xvar);
    else if (xstyle == ATOM)
      input->variable->compute_atom(xvar, igroup, &svelocity[0][0], 3, 0);
    if (ystyle == EQUAL)
      v_value[0][1] = input->variable->compute_equal(yvar);
    else if (ystyle == ATOM)
      input->variable->compute_atom(yvar, igroup, &svelocity[0][1], 3, 0);
    if (zstyle == EQUAL)
      v_value[0][2] = input->variable->compute_equal(zvar);
    else if (zstyle == ATOM)
      input->variable->compute_atom(zvar, igroup, &svelocity[0][2], 3, 0);
    
    modify->addstep_compute(update->ntimestep + 1);
    
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	if (region && !region->match(x[i][0], x[i][1], x[i][2]))
	  continue;
	
	// does not work for two phase systems
	C2A_ind = fac_p1_C2A*(x[i][1]-region->extent_ylo);
	
	foriginal[0] += f[i][0];
	foriginal[1] += f[i][1];
	foriginal[2] += f[i][2];
	if (xstyle == ATOM) {
	  v[i][0] = svelocity[i][0];
	  f[i][0] = 0.0;
	} else if (xstyle) {
	  v[i][0] = v_value[C2A_ind][0];
	  f[i][0] = 0.0;
	}
	
	if (ystyle == ATOM) {
	  v[i][1] = svelocity[i][1];
	  f[i][1] = 0.0;
	} else if (ystyle) {
	  v[i][1] = v_value[C2A_ind][1];
	  f[i][1] = 0.0;
	}
	
	if (zstyle == ATOM) {
	  v[i][2] = svelocity[i][2];
	  f[i][2] = 0.0;
	} else if (zstyle) {
	  v[i][2] = v_value[C2A_ind][2];
	  f[i][2] = 0.0;
	}
	
      }
  }
*/
}

void FixMDtoCFD::C_to_A_coupling_velocity(int vflag) {
  
  compute_spatial_averaged_velocity(phase1_region_c_to_a,phase1_groupbit);

  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  double *mass = atom->mass;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  
  // update region if necessary
  
  Region *region = NULL;
  if (phase1_region_c_to_a >= 0) {
    region = domain->regions[phase1_region_c_to_a];
    region->prematch();
  }

  // reallocate svelocity array if necessary
  
  if (varflag == ATOM && atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(svelocity);
    memory->create(svelocity, maxatom, 3, "setvelocity:svelocity");
  }

  foriginal[0] = foriginal[1] = foriginal[2] = 0.0;
  force_flag = 0;

  double tmp[nC2ADataPoints_p1][3];
  //if(update->ntimestep%nevery==0)
    for (int j=0; j< nC2ADataPoints_p1; j++){
      tmp[j][0] = (v_value[j][0]-phase1_C2A_global_avg_vel[j][0]);
      tmp[j][1] = (v_value[j][1]-phase1_C2A_global_avg_vel[j][1]);


      //fprintf(stdout,"TT %d %d %f %f \n",update->ntimestep,j,tmp[j][0],tmp[j][1]);
    }

  int C2A_ind=0;
  if (varflag == CONSTANT) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & phase1_groupbit) {
	if (region && !region->match(x[i][0], x[i][1], x[i][2]))
	  continue;
	
	// does not work for two phase systems
	C2A_ind = fac_p1_C2A*(x[i][1]-region->extent_ylo);
	
	if(C2A_ind<0 || C2A_ind>=nC2ADataPoints_p1) fprintf(stdout, "BIG ERROR %d %d\n", update->ntimestep, C2A_ind);
	
	//f[i][0] += mass[type[i]]*tmp[C2A_ind][0];
	//f[i][1] += mass[type[i]]*tmp[C2A_ind][1];
	
	foriginal[0] += f[i][0];
	foriginal[1] += f[i][1];
	foriginal[2] += f[i][2];

	if (xstyle) {
	  v[i][0] += tmp[C2A_ind][0];
	}
	if (ystyle) {	  
	  v[i][1] += tmp[C2A_ind][1];
	}
	if (zstyle) {
	  v[i][2] += tmp[C2A_ind][2];
	}
      }
 
    // variable force, wrap with clear/add
  }
  /*  else {
    
    modify->clearstep_compute();
    
    if (xstyle == EQUAL)
      v_value[0][0]= input->variable->compute_equal(xvar);
    else if (xstyle == ATOM)
      input->variable->compute_atom(xvar, igroup, &svelocity[0][0], 3, 0);
    if (ystyle == EQUAL)
      v_value[0][1] = input->variable->compute_equal(yvar);
    else if (ystyle == ATOM)
      input->variable->compute_atom(yvar, igroup, &svelocity[0][1], 3, 0);
    if (zstyle == EQUAL)
      v_value[0][2] = input->variable->compute_equal(zvar);
    else if (zstyle == ATOM)
      input->variable->compute_atom(zvar, igroup, &svelocity[0][2], 3, 0);
    
    modify->addstep_compute(update->ntimestep + 1);
    
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	if (region && !region->match(x[i][0], x[i][1], x[i][2]))
	  continue;
	
	// does not work for two phase systems
	C2A_ind = fac_p1_C2A*(x[i][1]-region->extent_ylo);
	
	foriginal[0] += f[i][0];
	foriginal[1] += f[i][1];
	foriginal[2] += f[i][2];
	if (xstyle == ATOM) {
	  v[i][0] = svelocity[i][0];
	  f[i][0] = 0.0;
	} else if (xstyle) {
	  v[i][0] = v_value[C2A_ind][0];
	  f[i][0] = 0.0;
	}
	
	if (ystyle == ATOM) {
	  v[i][1] = svelocity[i][1];
	  f[i][1] = 0.0;
	} else if (ystyle) {
	  v[i][1] = v_value[C2A_ind][1];
	  f[i][1] = 0.0;
	}
	
	if (zstyle == ATOM) {
	  v[i][2] = svelocity[i][2];
	  f[i][2] = 0.0;
	} else if (zstyle) {
	  v[i][2] = v_value[C2A_ind][2];
	  f[i][2] = 0.0;
	}
	
      }
  }
*/
}

void FixMDtoCFD::update_velocity(int vflag) {
    double **x = atom->x;
    double **f = atom->f;
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    // update region if necessary
    
    Region *region = NULL;
    if (phase1_region_c_to_a >= 0) {
        region = domain->regions[phase1_region_c_to_a];
	region->prematch();
    }

    // reallocate svelocity array if necessary

    if (varflag == ATOM && atom->nmax > maxatom) {
        maxatom = atom->nmax;
	memory->destroy(svelocity);
	memory->create(svelocity, maxatom, 3, "setvelocity:svelocity");
    }

    foriginal[0] = foriginal[1] = foriginal[2] = 0.0;
    force_flag = 0;

    int C2A_ind=0;
    if (varflag == CONSTANT) {
        for (int i = 0; i < nlocal; i++)
	    if (mask[i] & groupbit) {
	        if (region && !region->match(x[i][0], x[i][1], x[i][2]))
		    continue;
				
		// does not work for two phase systems
		C2A_ind = fac_p1_C2A*(x[i][1]-region->extent_ylo);

		foriginal[0] += f[i][0];
		foriginal[1] += f[i][1];
		foriginal[2] += f[i][2];
		
		if (xstyle) {
		  v[i][0] = v_value[C2A_ind][0];
		  //f[i][0] = 0.0;
		}
		if (ystyle) {
		  v[i][1] = v_value[C2A_ind][1];
		  //f[i][1] = 0.0;
		}
		if (zstyle) {
		  v[i][2] = v_value[C2A_ind][2];
		  //f[i][2] = 0.0;
		}
	    }

    // variable force, wrap with clear/add
    } else {
      
        modify->clearstep_compute();
	
	if (xstyle == EQUAL)
	    v_value[0][0]= input->variable->compute_equal(xvar);
	else if (xstyle == ATOM)
	    input->variable->compute_atom(xvar, igroup, &svelocity[0][0], 3, 0);
	if (ystyle == EQUAL)
	    v_value[0][1] = input->variable->compute_equal(yvar);
	else if (ystyle == ATOM)
	    input->variable->compute_atom(yvar, igroup, &svelocity[0][1], 3, 0);
	if (zstyle == EQUAL)
	    v_value[0][2] = input->variable->compute_equal(zvar);
	else if (zstyle == ATOM)
	    input->variable->compute_atom(zvar, igroup, &svelocity[0][2], 3, 0);
	
	modify->addstep_compute(update->ntimestep + 1);

	for (int i = 0; i < nlocal; i++)
	    if (mask[i] & groupbit) {
	        if (region && !region->match(x[i][0], x[i][1], x[i][2]))
		    continue;

		// does not work for two phase systems
		C2A_ind = fac_p1_C2A*(x[i][1]-region->extent_ylo);

		foriginal[0] += f[i][0];
		foriginal[1] += f[i][1];
		foriginal[2] += f[i][2];
		if (xstyle == ATOM) {
		    v[i][0] = svelocity[i][0];
		    f[i][0] = 0.0;
		} else if (xstyle) {
		    v[i][0] = v_value[C2A_ind][0];
		    f[i][0] = 0.0;
		}

		if (ystyle == ATOM) {
		    v[i][1] = svelocity[i][1];
		    f[i][1] = 0.0;
		} else if (ystyle) {
		    v[i][1] = v_value[C2A_ind][1];
		    f[i][1] = 0.0;
		}

		if (zstyle == ATOM) {
		    v[i][2] = svelocity[i][2];
		    f[i][2] = 0.0;
		} else if (zstyle) {
		    v[i][2] = v_value[C2A_ind][2];
		    f[i][2] = 0.0;
		}
		
	    }
    }
}

void FixMDtoCFD::update_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  if (lmp->kokkos)
    atom->sync_modify(Host, (unsigned int) (F_MASK | MASK_MASK),
                      (unsigned int) F_MASK);

  // update region if necessary

  Region *region = NULL;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }

  // reallocate sforce array if necessary

  if ((varflag == ATOM || estyle == ATOM) && nlocal > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(sforce);
    memory->create(sforce,maxatom,4,"hmm:sforce");
  }

  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added

  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  force_flag = 0;

  // constant force
  // potential energy = - x dot f in unwrapped coords

  if (varflag == CONSTANT) {
    double unwrap[3];
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        domain->unmap(x[i],image[i],unwrap);
        foriginal[0] -= f_xvalue*unwrap[0] + f_yvalue*unwrap[1] + f_zvalue*unwrap[2];
        foriginal[1] += f[i][0];
        foriginal[2] += f[i][1];
        foriginal[3] += f[i][2];
        f[i][0] += f_xvalue;
        f[i][1] += f_yvalue;
        f[i][2] += f_zvalue;
      }

  // variable force, wrap with clear/add
  // potential energy = evar if defined, else 0.0
  // wrap with clear/add

  } else {

    modify->clearstep_compute();

    if (xstyle == EQUAL) f_xvalue = input->variable->compute_equal(xvar);
    else if (xstyle == ATOM)
      input->variable->compute_atom(xvar,igroup,&sforce[0][0],4,0);
    if (ystyle == EQUAL) f_yvalue = input->variable->compute_equal(yvar);
    else if (ystyle == ATOM)
      input->variable->compute_atom(yvar,igroup,&sforce[0][1],4,0);
    if (zstyle == EQUAL) f_zvalue = input->variable->compute_equal(zvar);
    else if (zstyle == ATOM)
      input->variable->compute_atom(zvar,igroup,&sforce[0][2],4,0);
    if (estyle == ATOM)
      input->variable->compute_atom(evar,igroup,&sforce[0][3],4,0);

    modify->addstep_compute(update->ntimestep + 1);

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        if (estyle == ATOM) foriginal[0] += sforce[i][3];
        foriginal[1] += f[i][0];
        foriginal[2] += f[i][1];
        foriginal[3] += f[i][2];
        if (xstyle == ATOM) f[i][0] += sforce[i][0];
        else if (xstyle) f[i][0] += f_xvalue;
        if (ystyle == ATOM) f[i][1] += sforce[i][1];
        else if (ystyle) f[i][1] += f_yvalue;
        if (zstyle == ATOM) f[i][2] += sforce[i][2];
        else if (zstyle) f[i][2] += f_zvalue;
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixMDtoCFD::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixMDtoCFD::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixMDtoCFD::compute_scalar()
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return foriginal_all[0];
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixMDtoCFD::compute_vector(int n)
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return foriginal_all[n+1];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixMDtoCFD::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = maxatom*4 * sizeof(double);
  return bytes;
}

void FixMDtoCFD::compute_average_velocity(int ireg,int phase_groupbit)
{
  if (update->ntimestep<startavg) return;

  Region *region = NULL; 
  if (ireg >= 0) {
    region = domain->regions[ireg];
    region->prematch();
  }

  int A2C_ind=0;
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int j = 0; j < nA2CDataPoints_p1; j++){  
    phase1_A2C_numAtoms[j]=0;
  }

  for (int j = 0; j < nA2CDataPoints_p2; j++){  
    phase2_A2C_numAtoms[j]=0;
  }

  for (int i = 0; i < nlocal; i++){
    if (mask[i] & phase_groupbit) {
      if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;

      // does not work for two phase systems
      A2C_ind = fac_p1_A2C*(x[i][1]-region->extent_ylo);
      phase1_A2C_avg_vel_per_proc[A2C_ind][0]+=v[i][0];
      phase1_A2C_avg_vel_per_proc[A2C_ind][1]+=v[i][1];
      phase1_A2C_numAtoms[A2C_ind]++;
      if(A2C_ind<0 || A2C_ind>nA2CDataPoints_p1) fprintf(stdout,"ATOM %d %f!\n",A2C_ind,x[i][1]);
    }
  }
  /*
  for (int j = 0; j < nA2CDataPoints_p1; j++){
    phase1_avg_vel_per_proc[j][0]/=phase1_numAtoms[j];
    phase1_avg_vel_per_proc[j][1]/=phase1_numAtoms[j];
    //fprintf(stdout,"NO %d %d %d!\n",update->ntimestep,j,phase1_numAtoms[j]);
    }*/

  if (update->ntimestep >=startavg && update->ntimestep%nevery==0){
    double temp1=0;
    double temp2=0;
    int nAtoms=0;
    for(int j=0;j<nA2CDataPoints_p1;j++){
      MPI_Allreduce(&phase1_A2C_avg_vel_per_proc[j][0],&temp1,1,MPI_DOUBLE,MPI_SUM,world);
      MPI_Allreduce(&phase1_A2C_avg_vel_per_proc[j][1],&temp2,1,MPI_DOUBLE,MPI_SUM,world);
      MPI_Allreduce(&phase1_A2C_numAtoms[j],&nAtoms,1,MPI_INT,MPI_SUM,world);
      phase1_A2C_global_avg_vel[j][0]=temp1*factor_to_OF_vel/navgTime/nAtoms;
      phase1_A2C_global_avg_vel[j][1]=temp2*factor_to_OF_vel/navgTime/nAtoms; 
      fprintf(stdout,"A2C 1 %f %f %d\n",phase1_A2C_avg_vel_per_proc[j][1],phase1_A2C_global_avg_vel[j][1],navgTime);    
      phase1_A2C_avg_vel_per_proc[j][0]=0.0;
      phase1_A2C_avg_vel_per_proc[j][1]=0.0;
      phase1_nTimeZero[j]=0;
    }
    /*
    if(phase2_region_a_to_c>=0){
      for(int j=0;j<nA2CDataPoints_p2;j++){
	MPI_Allreduce(&phase2_avg_vel_per_proc[j],&phase2_global_avg_vel[j],1,MPI_DOUBLE,MPI_SUM,world);
	//phase2_global_avg_vel[j]=phase2_global_avg_vel[j]*factor_to_m_per_s/navgTime;
	phase2_avg_vel_per_proc[j][0]=0.0;
	phase2_avg_vel_per_proc[j][1]=0.0;
      }
    }
    */
    startavg=startavg+nevery;
    if (comm->me == 0)
      for(int j=0;j<nA2CDataPoints_p1;j++){
	fprintf(stdout,"current_step: %d  avg_vel_phase1: %f avg_vel_phase2: %f \n",update->ntimestep,phase1_A2C_global_avg_vel[j][0],phase2_A2C_global_avg_vel[j][1]);
      }
  }
}

void FixMDtoCFD::compute_spatial_averaged_velocity(int ireg, int phase_groupbit)
{
  Region *region = NULL; 
  if (ireg >= 0) {
    region = domain->regions[ireg];
    region->prematch();
  }

  int C2A_ind=0;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *type = atom->type;

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int j = 0; j < nC2ADataPoints_p1; j++){  
    phase1_C2A_numAtoms[j]=0;
    for(int k=0;k<3;k++)
    {
      phase1_C2A_avg_vel_per_proc[j][k]=0.0;
      phase1_C2A_avg_acc_per_proc[j][k]=0.0;
    }
  }

  for (int j = 0; j < nC2ADataPoints_p2; j++){  
    phase2_C2A_numAtoms[j]=0;
  }
  
  for (int i = 0; i < nlocal; i++){
    if (mask[i] & phase_groupbit) {
      if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;

      // does not work for two phase systems
      C2A_ind = fac_p1_C2A*(x[i][1]-region->extent_ylo);
      phase1_C2A_avg_vel_per_proc[C2A_ind][0]+=v[i][0];
      phase1_C2A_avg_vel_per_proc[C2A_ind][1]+=v[i][1];
      //phase1_C2A_avg_acc_per_proc[C2A_ind][0]+=f[i][0]/mass[type[i]];
      //phase1_C2A_avg_acc_per_proc[C2A_ind][1]+=f[i][1]/mass[type[i]];
      phase1_C2A_avg_acc_per_proc[C2A_ind][0]+=f[i][0];
      phase1_C2A_avg_acc_per_proc[C2A_ind][1]+=f[i][1];
      phase1_C2A_numAtoms[C2A_ind]++;
      if(C2A_ind<0 || C2A_ind>nC2ADataPoints_p1) fprintf(stdout,"ATOM %d %f!\n",C2A_ind,x[i][1]);
    }
  } 
  
  int nAtoms[nC2ADataPoints_p1];
  for(int j=0;j<nC2ADataPoints_p1;j++){
    MPI_Allreduce(phase1_C2A_avg_vel_per_proc[j],phase1_C2A_global_avg_vel[j],3,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(phase1_C2A_avg_acc_per_proc[j],phase1_C2A_global_avg_acc[j],3,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&phase1_C2A_numAtoms[j],&nAtoms[j],1,MPI_INT,MPI_SUM,world);
    for(int k=0;k<3;k++)
    {
      phase1_C2A_global_avg_vel[j][k]/=nAtoms[j];
      phase1_C2A_global_avg_acc[j][k]/=nAtoms[j];
    }
  }
  /*
    if(phase2_region_a_to_c>=0){
    for(int j=0;j<nA2CDataPoints_p2;j++){
    MPI_Allreduce(&phase2_avg_vel_per_proc[j],&phase2_global_avg_vel[j],1,MPI_DOUBLE,MPI_SUM,world);
    //phase2_global_avg_vel[j]=phase2_global_avg_vel[j]*factor_to_m_per_s/navgTime;
    phase2_avg_vel_per_proc[j][0]=0.0;
    phase2_avg_vel_per_proc[j][1]=0.0;
    }
    }
  */
  
  if (comm->me == 0 && update->ntimestep%nevery==0)
    for(int j=0;j<nC2ADataPoints_p1;j++){
      fprintf(stdout,"current_step: %d  avg_vel_phase1: %f avg_vel_phase2: %f %f %f %f %f %d %d\n",update->ntimestep,phase1_C2A_global_avg_vel[j][0],phase2_C2A_global_avg_vel[j][1], v_value[j][0], v_value[j][1],phase1_C2A_global_avg_acc[j][0],phase2_C2A_global_avg_acc[j][1],j,nAtoms[j]);
    }
}

void FixMDtoCFD::compute_time_averaged_velocity(int ireg)
{
  int i,j,m,n,index;
  int nvalues=3;
  int nevery=1;

  // skip if not step which requires doing something
  // error check if timestep was reset in an invalid manner
  bigint ntimestep = update->ntimestep;
  if (ntimestep < nvalid_last || ntimestep > nvalid)
    error->all(FLERR,"Invalid timestep reset for fix ave/chunk");
  if (ntimestep != nvalid) return;
  nvalid_last = nvalid;

  // first sample within single Nfreq epoch
  // zero out arrays that accumulate over many samples, but not across epochs
  // invoke setup_chunks() to determine current nchunk
  //   re-allocate per-chunk arrays if needed
  // invoke lock() in two cases:
  //   if nrepeat > 1: so nchunk cannot change until Nfreq epoch is over,
  //     will be unlocked on last repeat of this Nfreq
  //   if ave = RUNNING/WINDOW and not yet locked:
  //     set forever, will be unlocked in fix destructor
  // wrap setup_chunks in clearstep/addstep b/c it may invoke computes
  //   both nevery and nfreq are future steps,
  //   since call below to cchunk->ichunk()
  //     does not re-invoke internal cchunk compute on this same step

  if (irepeat == 0) {
    if (cchunk->computeflag) modify->clearstep_compute();
    nchunk = cchunk->setup_chunks();
    if (cchunk->computeflag) {
      modify->addstep_compute(ntimestep+nevery);
      modify->addstep_compute(ntimestep+nfreq);
    }
    
    allocate();
    
    if (nrepeat > 1)
      cchunk->lock(this,ntimestep,ntimestep+(nrepeat-1)*nevery);
    for (m = 0; m < nchunk; m++) {
      count_many[m] = count_sum[m] = 0.0;
      for (i = 0; i < nvalues; i++) values_many[m][i] = 0.0;
    }
    
  }
  
  // zero out arrays for one sample
  for (m = 0; m < nchunk; m++) {
    count_one[m] = 0.0;
    for (i = 0; i < nvalues; i++) values_one[m][i] = 0.0;
  }
  
  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms
  // wrap compute_ichunk in clearstep/addstep b/c it may invoke computes

  if (cchunk->computeflag) modify->clearstep_compute();

  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if (cchunk->computeflag) modify->addstep_compute(ntimestep+nevery);

  // perform the computation for one sample
  // count # of atoms in each bin
  // accumulate velocities to local copy
  // sum within each chunk, only include atoms in fix group
  // compute/fix/variable may invoke computes so wrap with clear/add
  
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit && ichunk[i] > 0)
      count_one[ichunk[i]-1]++;

  modify->clearstep_compute();
  
  for (m = 0; m < nvalues; m++) {
  // adds velocities to values    
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit && ichunk[i] > 0) {
	index = ichunk[i]-1;
	values_one[index][m] += atom->v[i][m];
      }
  }
  
  // accumulate values,count separately to many
  {
    for (m = 0; m < nchunk; m++) {
      count_many[m] += count_one[m];
      for (j = 0; j < nvalues; j++)
        values_many[m][j] += values_one[m][j];
    }
  }
  
  // done if irepeat < nrepeat
  // else reset irepeat and nvalid
  irepeat++;
  if (irepeat < nrepeat) {
    nvalid += nevery;
    modify->addstep_compute(nvalid); 
    return;
  }
  
  irepeat = 0;
  nvalid = ntimestep+nfreq - (nrepeat-1)*nevery;
  modify->addstep_compute(nvalid);

  // unlock compute chunk/atom at end of Nfreq epoch
  // do not unlock if ave = RUNNING or WINDOW

  if (nrepeat > 1) cchunk->unlock(this);

  // time average is total value / total count
  double repeat = nrepeat;
  
  MPI_Allreduce(count_many,count_sum,nchunk,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&values_many[0][0],&values_sum[0][0],nchunk*nvalues,
		MPI_DOUBLE,MPI_SUM,world);
  
  for (m = 0; m < nchunk; m++) {
    if (count_sum[m] > 0.0)
      for (j = 0; j < nvalues; j++) {
	values_sum[m][j] /= count_sum[m];
      }
    count_sum[m] /= repeat;
  }

  for (m = 0; m < nchunk; m++) {
    for (i = 0; i < nvalues; i++)
      values_total[m][i] = values_sum[m][i];
    count_total[m] = count_sum[m];
  }

  Region *region = NULL; 
  if (ireg >= 0) {
    region = domain->regions[ireg];
    region->prematch();
  }
  
  double **coord = cchunk->coord; 
  double chunksize=(coord[nchunk-1][0]-coord[0][0])/(nchunk-1);
  double regsize=(region->extent_yhi-region->extent_ylo)/nA2CDataPoints_p1;

  for (int j = 0; j < nA2CDataPoints_p1; j++) {
    double regC=region->extent_ylo+(j+1)*regsize*0.5;
    int ind1=floor((regC-coord[0][0])/chunksize);
    int ind2=ceil((regC-coord[0][0])/chunksize);
  
    phase1_A2C_global_avg_vel[j][0]=values_total[ind1][0]+(regC-coord[ind1][0])*(values_total[ind2][0]-values_total[ind1][0])/(coord[ind2][0]-coord[ind1][0]);
    phase1_A2C_global_avg_vel[j][1]=values_total[ind1][1]+(regC-coord[ind1][0])*(values_total[ind2][1]-values_total[ind1][1])/(coord[ind2][0]-coord[ind1][0]);
    phase1_A2C_global_avg_vel[j][0]*=factor_to_OF_vel;
    phase1_A2C_global_avg_vel[j][1]*=factor_to_OF_vel;
    fprintf(stdout,"phase1_vel_coord %d %d %f %f %f %f %f\n", update->ntimestep,j,regC,phase1_A2C_global_avg_vel[j][0],phase1_A2C_global_avg_vel[j][1],phase1_A2C_global_avg_vel[j][0]/factor_to_OF_vel,phase1_A2C_global_avg_vel[j][1]/factor_to_OF_vel);
    //double d0=values_total[ind1][0]+(regC-coord[ind1][0])*(values_total[ind2][0]-values_total[ind1][0])/(coord[ind2][0]-coord[ind1][0]);
    //double d1=values_total[ind1][1]+(regC-coord[ind1][0])*(values_total[ind2][1]-values_total[ind1][1])/(coord[ind2][0]-coord[ind1][0]);
    //d0*=factor_to_OF_vel;
    //d1*=factor_to_OF_vel;
    //fprintf(stdout,"phase1_vel_coord %d %d %f %f %f \n", update->ntimestep,j,regC,d0,d1);
  }

  // output result to file
  fp=fopen("timeaveraged_velocity.txt","a");
  if (fp && comm->me == 0) {
    clearerr(fp);
    double count = 0.0;
    for (m = 0; m < nchunk; m++) count += count_total[m];
    fprintf(fp,BIGINT_FORMAT " %d %g\n",ntimestep,nchunk,count);

    int compress = cchunk->compress;
    int *chunkID = cchunk->chunkID;
    int ncoord = cchunk->ncoord;
    double **coord = cchunk->coord;    

    if (!compress) {
      if (ncoord == 1) {
        for (m = 0; m < nchunk; m++) {
          fprintf(fp,"  %d %g %g",m+1,coord[m][0],
                  count_total[m]);
          for (i = 0; i < nvalues; i++)
            fprintf(fp,format,values_total[m][i]);
          fprintf(fp,"\n");
        }
      } 
    } 
    fflush(fp);
  }
  fclose(fp);

}
/* ----------------------------------------------------------------------
   calculate nvalid = next step on which end_of_step does something
   can be this timestep if multiple of nfreq and nrepeat = 1
   else backup from next multiple of nfreq
------------------------------------------------------------------------- */

bigint FixMDtoCFD::nextvalid()
{
  int nevery=1;
  bigint nvalid = (update->ntimestep/nfreq)*nfreq + nfreq;
  if (nvalid-nfreq == update->ntimestep && nrepeat == 1)
    nvalid = update->ntimestep;
  else
    nvalid -= (nrepeat-1)*nevery;

  if (nvalid < update->ntimestep) nvalid += nfreq;

  return nvalid;
}

/* ----------------------------------------------------------------------
   allocate all per-chunk vectors
------------------------------------------------------------------------- */

void FixMDtoCFD::allocate()
{
  int nvalues=3;
  size_array_rows = nchunk;

  // reallocate chunk arrays if needed

  if (nchunk > maxchunk) {
    maxchunk = nchunk;
    memory->grow(count_one,nchunk,"ave/chunk:count_one");
    memory->grow(count_many,nchunk,"ave/chunk:count_many");
    memory->grow(count_sum,nchunk,"ave/chunk:count_sum");
    memory->grow(count_total,nchunk,"ave/chunk:count_total");

    memory->grow(values_one,nchunk,nvalues,"ave/chunk:values_one");
    memory->grow(values_many,nchunk,nvalues,"ave/chunk:values_many");
    memory->grow(values_sum,nchunk,nvalues,"ave/chunk:values_sum");
    memory->grow(values_total,nchunk,nvalues,"ave/chunk:values_total");

    // reinitialize regrown count/values total since they accumulate

    int i,m;
    for (m = 0; m < nchunk; m++) {
      for (i = 0; i < nvalues; i++) values_total[m][i] = 0.0;
      count_total[m] = 0.0;
    }
  }
}

