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

/* ----------------------------------------------------------------------
   Contributing author: Srinivasa Ramisetti (University of Edinburgh)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_coul_smooth.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairCoulSmooth::PairCoulSmooth(LAMMPS *lmp) : Pair(lmp) {}

/* ---------------------------------------------------------------------- */

PairCoulSmooth::~PairCoulSmooth()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(scale);
    memory->destroy(cutsqinv);
    memory->destroy(cutinv);
  }
}

/* ---------------------------------------------------------------------- */

void PairCoulSmooth::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,ecoul,fpair;
  double rsq,r2inv,rinv,forcecoul,factor_coul,rcut,rcut2inv,dist;
  int *ilist,*jlist,*numneigh,**firstneigh;

  ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int ncoul=0;
  // FILE *fp;
  // fp = fopen("pcoul.txt", "a+");
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

		rcut=cut[itype][jtype];
        rcut2inv = cutsqinv[itype][jtype];
		//dist=(1.0/rinv)-rcut;
        
		// forcecoul = qqrd2e * scale[itype][jtype] * qtmp*q[j]*rinv;
		// fpair = factor_coul*forcecoul * r2inv;

        //forcecoul = qqrd2e * scale[itype][jtype] * qtmp*q[j]*(rinv-cutinv[itype][jtype]-dist*rcut2inv);
		forcecoul = qqrd2e * scale[itype][jtype] * qtmp*q[j]*rinv;
		fpair = factor_coul*forcecoul * (r2inv-1.0/cut_global/cut_global);

		//if (itype!=2 && jtype!=2) ncoul++;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag){
			//ecoul = factor_coul * qqrd2e * scale[itype][jtype] * qtmp*q[j]*rinv;
			ecoul = factor_coul * qqrd2e * scale[itype][jtype] * qtmp*q[j] * ( rinv - (2.0/cut_global) + (sqrt(rsq)/cut_global/cut_global) );

//			(rinv-2.0*cutinv[itype][jtype]+sqrt(rsq)*cutsqinv[itype][jtype])
//			fprintf(fp,"NCOUL %d %d %e %e %e\n",i,j,qtmp,q[j],ecoul);
			//fprintf(fp,"NCOUL %d %d %e %e %e %e %e %e %e %e %e %e\n",i,j,qtmp,q[j],x[i][0],x[i][1],x[i][2],x[j][0],x[j][1],x[j][2],sqrt(rsq),ecoul);
		}
        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             0.0,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  // fprintf(fp,"\n\n");
  // fclose(fp);
  // fprintf(stdout,"NCOUL %d\n",ncoul);
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairCoulSmooth::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(scale,n+1,n+1,"pair:scale");
  memory->create(cutsqinv,n+1,n+1,"pair:cutsqinv");
  memory->create(cutinv,n+1,n+1,"pair:cutinv");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCoulSmooth::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairCoulSmooth::coeff(int narg, char **arg)
{
  if (narg < 2 || narg > 3)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double cut_one = cut_global;
  if (narg == 3) cut_one = force->numeric(FLERR,arg[2]);

  int count = 0;
  double cut_one_inv=1.0/cut_one;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut[i][j] = cut_one;
	  cutinv[i][j] = cut_one_inv;
	  cutsqinv[i][j] = cut_one_inv*cut_one_inv;
      scale[i][j] = 1.0;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairCoulSmooth::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style coul/cut requires atom attribute q");

  neighbor->request(this,instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairCoulSmooth::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);

  scale[j][i] = scale[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCoulSmooth::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) fwrite(&cut[i][j],sizeof(double),1,fp);
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCoulSmooth::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) fread(&cut[i][j],sizeof(double),1,fp);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCoulSmooth::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCoulSmooth::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairCoulSmooth::single(int i, int j, int itype, int jtype,
                           double rsq, double factor_coul, double factor_lj,
                           double &fforce)
{
  double r2inv,rinv,forcecoul,phicoul;

  r2inv = 1.0/rsq;
  rinv = sqrt(r2inv);
  //forcecoul = force->qqrd2e * atom->q[i]*atom->q[j]*rinv;
  //fforce = factor_coul*forcecoul * r2inv;

  forcecoul = force->qqrd2e * atom->q[i]*atom->q[j]*(rinv-2.0*cutinv[itype][jtype]+sqrt(rsq)*cutsqinv[itype][jtype]);
  fforce = factor_coul*forcecoul * (r2inv-cutsqinv[itype][jtype]);

  //phicoul = force->qqrd2e * atom->q[i]*atom->q[j]*rinv;
  phicoul = force->qqrd2e * atom->q[i]*atom->q[j]*(rinv-2.0*cutinv[itype][jtype]+sqrt(rsq)*cutsqinv[itype][jtype]);
  return factor_coul*phicoul;
}

/* ---------------------------------------------------------------------- */

void *PairCoulSmooth::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"scale") == 0) return (void *) scale;
  return NULL;
}
