#!/bin/bash

sh train_di_vae_slurm.sh --exclude_domains WEATHER_CHECK
sleep 5
sh train_di_vae_slurm.sh --exclude_domains UPDATE_CALENDAR APPOINTMENT_REMINDER
sleep 5
sh train_di_vae_slurm.sh --exclude_domains CITY_INFO
