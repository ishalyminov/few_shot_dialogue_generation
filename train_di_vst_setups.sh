#!/bin/bash

sbatch train_di_vst_slurm.sh --exclude_domains WEATHER_CHECK
sleep 5
sbatch train_di_vst_slurm.sh --exclude_domains UPDATE_CALENDAR APPOINTMENT_REMINDER
sleep 5
sbatch train_di_vst_slurm.sh --exclude_domains CITY_INFO
