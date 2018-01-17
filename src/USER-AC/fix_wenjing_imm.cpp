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
#include "fix_wenjing_imm.h"
#include "atom.h"
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

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixWenjingIMM::FixWenjingIMM(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
#ifdef ENABLE_MUI
  std::string appName_ = arg[3];
  std::string ifsName = arg[4];
  ifsName="ifs"+ifsName;
  interfaceNames.push_back(ifsName);
  interfaces_ = mui::create_uniface<mui::config_1d>(appName_, interfaceNames);
  imm_nIter=0;
#endif
  factor_to_kg_per_s=(1e-3*1e+15)/6.022140857e23;

  if (narg < 5) error->all(FLERR,"Illegal fix imm command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  velocity_x_per_proc=0.0;
  velocity_y_per_proc=0.0;
  number_per_proc=0.0;
  global_x_velocity=0.0;
  global_y_velocity=0.0;

  eta[0]=0.1;//0.02;
  eta[1]=0.1;//0.02;
  eta[2]=0.1;//0.02;
  
  boltz = 1.3806504e-23; //J/K
  sigma_lj = 0.34e-9;    //m
  epsilon_lj = 1.67e-21; //Joules
  mass_lj = 6.63e-26;    //kg

  cfd2md_L = 1.0/sigma_lj;
  cfd2md_time = sqrt(epsilon_lj/mass_lj)/sigma_lj;
  cfd2md_temp = boltz/epsilon_lj;
  cfd2md_v = cfd2md_L/cfd2md_time; 
  cfd2md_rho = (sigma_lj*sigma_lj*sigma_lj)/mass_lj;

  L_md2cfd = sigma_lj;
  time_md2cfd =1.0/cfd2md_time;
  temp_md2cfd = 1.0/cfd2md_temp;
  v_md2cfd = 1.0/cfd2md_v;
  rho_md2cfd = 1.0/cfd2md_rho;
	
  
  xstr = ystr = zstr = NULL;

  xvalue=0;
  xstyle = CONSTANT;

  yvalue=0;
  ystyle = CONSTANT;

  zvalue=0;
  zstyle = CONSTANT;

  // optional args

  nevery = 1;
  iregion1 = -1;
  iregion2 = -1;
  idregion1 = NULL;
  idregion2 = NULL;
  estr = NULL;

  startavg=0;
  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix imm command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix imm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"startavg") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix imm command");
      startavg = atoi(arg[iarg+1]);
      if (startavg <= 0) error->all(FLERR,"Illegal fix imm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"regionC2A") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix imm command");
      iregion1 = domain->find_region(arg[iarg+1]);
      if (iregion1 == -1)
        error->all(FLERR,"Region C to A ID for fix imm does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion1 = new char[n];
      strcpy(idregion1,arg[iarg+1]);
      iarg += 2;
	} else if (strcmp(arg[iarg],"regionA2C") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix imm command");
      iregion2 = domain->find_region(arg[iarg+1]);
      if (iregion2 == -1)
        error->all(FLERR,"Region A to C ID for fix imm does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion2 = new char[n];
      strcpy(idregion2,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix imm command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        int n = strlen(&arg[iarg+1][2]) + 1;
        estr = new char[n];
        strcpy(estr,&arg[iarg+1][2]);
      } else error->all(FLERR,"Illegal fix imm command");
      iarg += 2;
    } else if(strcmp(arg[iarg],"streamwidth") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix imm command");
      stream_width = atof(arg[iarg+1]);
      if (stream_width <= 0) error->all(FLERR,"Illegal fix imm command");
      iarg += 2;
    }
	else error->all(FLERR,"Illegal fix imm command");
  }

  navgTime=nevery-startavg;
  force_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;

  if(stream_width==0)
	  error->all(FLERR,"fix imm command needs streamwidth to be defined");

  maxatom = 1;
  memory->create(sforce,maxatom,4,"imm:sforce");
}

/* ---------------------------------------------------------------------- */

FixWenjingIMM::~FixWenjingIMM()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  delete [] estr;
  delete [] idregion1;
  delete [] idregion2;
#ifdef _MUI
  for(size_t i=0; interfaces_.size(); i++)
	  delete interfaces_[i];
#endif
  memory->destroy(sforce);
}

/* ---------------------------------------------------------------------- */

int FixWenjingIMM::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= FINAL_INTEGRATE;
  mask |= POST_FORCE;
  mask |= END_OF_STEP;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWenjingIMM::init()
{
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix imm does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
    else error->all(FLERR,"Variable for fix imm is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix imm does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
    else error->all(FLERR,"Variable for fix imm is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix imm does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
    else error->all(FLERR,"Variable for fix imm is invalid style");
  }
  if (estr) {
    evar = input->variable->find(estr);
    if (evar < 0)
      error->all(FLERR,"Variable name for fix imm does not exist");
    if (input->variable->atomstyle(evar)) estyle = ATOM;
    else error->all(FLERR,"Variable for fix imm is invalid style");
  } else estyle = NONE;

  // set index and check validity of region

  if (iregion1 >= 0) {
    iregion1 = domain->find_region(idregion1);
    if (iregion1 == -1)
      error->all(FLERR,"Region C to A ID for fix imm does not exist");
  }
  
  if (iregion2 >= 0) {
    iregion2 = domain->find_region(idregion2);
    if (iregion2 == -1)
      error->all(FLERR,"Region A to C ID for fix imm does not exist");
  }
  

  if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

  if (varflag == CONSTANT && estyle != NONE)
    error->all(FLERR,"Cannot use variable energy with "
               "constant force in fix imm");
  if ((varflag == EQUAL || varflag == ATOM) &&
      update->whichflag == 2 && estyle == NONE)
    error->all(FLERR,"Must use variable energy with fix imm");

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
		 mui::point1d loc(lc);		  
	  	 xvalue=cfd2md_v*interfaces_[i]->fetch( "velocity_x_", loc, imm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() ); 
		 yvalue=0;//cfd2md_v*interfaces_[i]->fetch( "velocity_y_", loc, imm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() ); 
		 if (comm->me == 0)
			 fprintf(stdout,"Initial LAMMPS fetch X velocity: %f, Y velocity: %f, ifs name: %s location: %d \n", xvalue, yvalue, interfaces_[i]->getIFS().c_str(), lc);
	  }
#endif

}

/* ---------------------------------------------------------------------- */

void FixWenjingIMM::final_integrate()
{
/*  double dtfm;

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
  }
*/
   if (update->ntimestep >=startavg){
	   velocity_x_per_proc+=compute_velocity_x();
	   velocity_y_per_proc+=compute_velocity_y();
	   number_per_proc+=compute_atom_number();
   }
	   
  if (update->ntimestep >=startavg && update->ntimestep%nevery==0){
	  MPI_Allreduce(&velocity_x_per_proc,&global_x_velocity,1,MPI_DOUBLE,MPI_SUM,world);
          MPI_Allreduce(&velocity_y_per_proc,&global_y_velocity,1,MPI_DOUBLE,MPI_SUM,world);
	  MPI_Allreduce(&number_per_proc,&global_number,1,MPI_DOUBLE,MPI_SUM,world);
	  global_x_velocity=global_x_velocity/global_number;
	  global_y_velocity=global_y_velocity/global_number;
	  velocity_x_per_proc=0.0;
          velocity_y_per_proc=0.0;
	  number_per_proc=0.0;
	  startavg=startavg+nevery;
	  if (comm->me == 0)
		  fprintf(stdout,"current_step: %d  averaged X velocity: %f,  Y velocity: %f \n",update->ntimestep,global_x_velocity,global_y_velocity);
  }
}

/* ---------------------------------------------------------------------- */

void FixWenjingIMM::end_of_step()
{
	
	
#ifdef ENABLE_MUI
  if (update->ntimestep%nevery==0){
	  for ( size_t i =  0; i <  interfaces_.size(); i++ ) 
	  {
		  int lc=std::stoi(interfaces_[i]->getIFS().substr(3,1));
		  mui::point1d loc(lc);
		  global_x_velocity=global_x_velocity*v_md2cfd;
		  global_y_velocity=global_y_velocity*v_md2cfd;
		  interfaces_[i]->push( "md_averaged_velocity_x_", loc, global_x_velocity);
		  interfaces_[i]->push( "md_averaged_velocity_y_", loc, global_y_velocity);
		  if (comm->me == 0)
			  fprintf(stdout,"LAMMPS push averaged X velocity: %f, Y velocity: %f,  ifs name: %s,  location: %d \n", global_x_velocity, global_y_velocity, interfaces_[i]->getIFS().c_str(), lc);
		  interfaces_[i]->commit(imm_nIter);
	  }
	  imm_nIter++;

	  if (comm->me == 0){
		  std::ofstream myfile;
		  myfile.open (interfaces_[0]->getIFS(), std::ios::out | std::ios::app);
		  if(update->ntimestep==nevery)
			  myfile << "IFS: " << interfaces_[0]->getIFS() << "\n";
		  myfile << "time " << update->ntimestep << " CFD_velocity: " << xvalue << " MD_velocity_X: " << global_x_velocity << " MD_velocity_Y: " << global_y_velocity << "\n";
		  myfile.close();
	  }

	  if(update->ntimestep!=update->laststep){
		for ( size_t i =  0; i <  interfaces_.size(); i++ ) 
		{
		  std::string lb_x="velocity_x_";
		  std::string lb_y="velocity_y_";
		  int lc=std::stoi(interfaces_[i]->getIFS().substr(3,1));
		  mui::point1d loc(lc);
		  xvalue=cfd2md_v*interfaces_[i]->fetch( lb_x, loc, imm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() ); 
		  yvalue=0;//cfd2md_v*interfaces_[i]->fetch( lb_y, loc, imm_nIter, mui::sampler_exact1d<double>(), mui::chrono_sampler_exact1d() ); 
	  	  if (comm->me == 0)
			  fprintf(stdout,"LAMMPS fetch X velocity: %f, Y velocity: %f, ifs name: %s location: %d \n", xvalue, yvalue, interfaces_[i]->getIFS().c_str(), lc);
		}
	  }

  }
#endif  

	//C_to_A_scaling();
	
}

/* ---------------------------------------------------------------------- */

void FixWenjingIMM::setup(int vflag)
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

void FixWenjingIMM::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWenjingIMM::post_force(int vflag)
{
	C_to_A_coupling();
}

/* ---------------------------------------------------------------------- */

void FixWenjingIMM::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWenjingIMM::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixWenjingIMM::compute_scalar()
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

double FixWenjingIMM::compute_vector(int n)
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

double FixWenjingIMM::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = maxatom*4 * sizeof(double);
  return bytes;
}

double FixWenjingIMM::compute_velocity_x()
{
  double velocity=0.0;
  double **v = atom->v;
  double **x = atom->x;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  
  Region *region2 = NULL;
  if (iregion2 >= 0) {
	region2 = domain->regions[iregion2];
	region2->prematch();
  }
	
  int arC=0;
  if (rmass) {
	for (int i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit && region2->match(x[i][0],x[i][1],x[i][2])) {
		velocity+=v[i][0];
	  }
		
  } else {
	  for (int i = 0; i < nlocal; i++)
		if (mask[i] & groupbit && region2->match(x[i][0],x[i][1],x[i][2])) {
		  velocity+=v[i][0];
		  arC++;
		}
  }

  return velocity;
}

double FixWenjingIMM::compute_velocity_y()
{
  double velocity=0.0;
  double **v = atom->v;
  double **x = atom->x;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  
  Region *region2 = NULL;
  if (iregion2 >= 0) {
	region2 = domain->regions[iregion2];
	region2->prematch();
  }
	
  int arC=0;
  if (rmass) {
	for (int i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit && region2->match(x[i][0],x[i][1],x[i][2])) {
		velocity+=v[i][1];
	  }
		
  } else {
	  for (int i = 0; i < nlocal; i++)
		if (mask[i] & groupbit && region2->match(x[i][0],x[i][1],x[i][2])) {
		  velocity+=v[i][1];
		  arC++;
		}
  }

  return velocity;
}

double FixWenjingIMM::compute_atom_number()
{
  double number=0.0;
  double **v = atom->v;
  double **x = atom->x;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  
  Region *region2 = NULL;
  if (iregion2 >= 0) {
	region2 = domain->regions[iregion2];
	region2->prematch();
  }
	
  if (rmass) {
	for (int i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit && region2->match(x[i][0],x[i][1],x[i][2])) {
		    number+=1.0;
	  }
		
  } else {
	  for (int i = 0; i < nlocal; i++)
		if (mask[i] & groupbit && region2->match(x[i][0],x[i][1],x[i][2])) {
		    number+=1.0;
		}
  }

  return number;
}

void FixWenjingIMM::C_to_A_scaling()
{

	double **v = atom->v;
	double **x = atom->x;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;

	double md_v_current_local[3];
	double md_v_current_global[3];
	int bincounter_local;
	int bincounter_global;
	
	bincounter_local = 0;
	bincounter_global = 0;
		
	for (int i=0; i<3; i++) {
		md_v_current_local[i] = 0.0;
		md_v_current_global[i] = 0.0;
	}

	Region *region1 = NULL;
	if (iregion1 >= 0) {
		region1 = domain->regions[iregion1];
		region1->prematch();
	}

	//find local total velocities and count how many atoms in C->A region
	for (int n=0; n<nlocal; n++) {
		if(mask[n] & groupbit && region1->match(x[n][0],x[n][1],x[n][2])) {
			md_v_current_local[0] += v[n][0];
			md_v_current_local[1] += v[n][1];
			md_v_current_local[2] += v[n][2];
			bincounter_local++;
		}//mask & region
	}//nlocal


	//reduce to find global velocities
	MPI_Allreduce(&md_v_current_local, &md_v_current_global, 3, MPI_DOUBLE, 
	              MPI_SUM, world);
	MPI_Allreduce(&bincounter_local, &bincounter_global, 1, MPI_INT, MPI_SUM, world);
	
	
	double d_in_bin; //total number of atoms in C->A region (double)
	double v_avg_bin[3]; //avg velocity of atom in C->A region
	double scale_factor[3]; //factor to scale velocities by (for momentum matching)

	//scaling...
	if(bincounter_global!=0) {	
		d_in_bin = static_cast<double>(bincounter_global);
		v_avg_bin[0] = md_v_current_global[0]/d_in_bin;
		v_avg_bin[1] = md_v_current_global[1]/d_in_bin;
		v_avg_bin[2] = md_v_current_global[2]/d_in_bin;
		scale_factor[0] = eta[0]*(xvalue - v_avg_bin[0]);
		scale_factor[1] = eta[1]*(yvalue - v_avg_bin[1]);
		//scale_factor[2] = eta[2]*(zvalue - v_avg_bin[2]);

		for (int n=0; n<nlocal; n++) {
			if(mask[n] & groupbit && region1->match(x[n][0],x[n][1],x[n][2])) {
				v[n][0]+=scale_factor[0];
				v[n][1]+=scale_factor[1];
				//v[n][2]+=scale_factor[2];
			}//mask
		}//nlocal
	}//bincounter check

}//end FixWenjingIMM::C_to_A_scaling()


void FixWenjingIMM::C_to_A_coupling()
{

	double **v = atom->v;
	double **x = atom->x;
	double **f = atom->f;
	double *mass = atom->mass;
	int *mask = atom->mask;
	int *type = atom->type;
	int nlocal = atom->nlocal;

	double md_v_current_local[3];
	double md_v_current_global[3];
	double md_a_current_local[3];
	double md_a_current_global[3];
	int bincounter_local;
	int bincounter_global;
	
	bincounter_local = 0;
	bincounter_global = 0;
		
	for (int i=0; i<3; i++) {
		md_v_current_local[i] = 0.0;
		md_v_current_global[i] = 0.0;
		md_a_current_local[i] = 0.0;
		md_a_current_global[i] = 0.0;
	}

	Region *region1 = NULL;
	if (iregion1 >= 0) {
		region1 = domain->regions[iregion1];
		region1->prematch();
	}

	//find local total velocities and count how many atoms in C->A region
	for (int n=0; n<nlocal; n++) {
		if(mask[n] & groupbit && region1->match(x[n][0],x[n][1],x[n][2])) {
			md_v_current_local[0] += v[n][0];
			md_v_current_local[1] += v[n][1];
			md_v_current_local[2] += v[n][2];
			md_a_current_local[0] += f[n][0]/mass[type[n]];
			md_a_current_local[1] += f[n][1]/mass[type[n]];
			md_a_current_local[2] += f[n][2]/mass[type[n]];
			bincounter_local++;
		}//mask & region
	}//nlocal


	//reduce to find global velocities
	MPI_Allreduce(&md_v_current_local, &md_v_current_global, 3, MPI_DOUBLE, 
	              MPI_SUM, world);
	MPI_Allreduce(&md_a_current_local, &md_a_current_global, 3, MPI_DOUBLE, 
	              MPI_SUM, world);
	MPI_Allreduce(&bincounter_local, &bincounter_global, 1, MPI_INT, MPI_SUM, world);
	
	
	double d_in_bin; //total number of atoms in C->A region (double)
	double v_avg_bin[3]; //avg velocity of atom in C->A region
	double a_avg_bin[3]; //avg acceleration of atom in C->A region
	double scale_factor[3]; //factor to scale velocities by (for momentum matching)

	//scaling...
	if(bincounter_global!=0) {	
		d_in_bin = static_cast<double>(bincounter_global);
		v_avg_bin[0] = md_v_current_global[0]/d_in_bin;
		v_avg_bin[1] = md_v_current_global[1]/d_in_bin;
		v_avg_bin[2] = md_v_current_global[2]/d_in_bin;
		a_avg_bin[0] = md_a_current_global[0]/d_in_bin;
		a_avg_bin[1] = md_a_current_global[1]/d_in_bin;
		a_avg_bin[2] = md_a_current_global[2]/d_in_bin;
		scale_factor[0] = eta[0]*(xvalue - v_avg_bin[0])/update->dt - a_avg_bin[0];
		scale_factor[1] = eta[1]*(yvalue - v_avg_bin[1])/update->dt - a_avg_bin[1];
		//scale_factor[2] = eta[2]*(zvalue - v_avg_bin[2])/update->dt - a_avg_bin[2];

		for (int n=0; n<nlocal; n++) {
			if(mask[n] & groupbit && region1->match(x[n][0],x[n][1],x[n][2])) {
				f[n][0]+=scale_factor[0];
				f[n][1]+=scale_factor[1];
				//f[n][2]+=scale_factor[2];
			}//mask
		}//nlocal
	}//bincounter check

}//end FixWenjingIMM::C_to_A_coupling()
