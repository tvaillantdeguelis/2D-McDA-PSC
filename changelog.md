**Version 2.7.0** Released on 2026-06-08.

* Added generation of ODL metadata (.met) files.

**Version 2.6.3** Released on 2026-06-04.

* Changed typical_range for HNO3 mixing ratio.

**Version 2.6.2** Released on 2026-06-01.

* Renamed 'valid_range' by 'typical_range' in the metadata.

**Version 2.6.1** Released on 2026-05-23.

* Optimized NetCDF output variable formats.

**Version 2.6.0** Released on 2026-05-19.

* Replaced the fixed 215 hPa threshold with the MERRA-2 tropopause to flag "Likely tropospheric features".

**Version 2.5.0** Released on 2026-05-19.

* Remove '_lon_X_X' in filename when all lat > 50° processed.

* Fixed: Replaced pyplot stateful calls with explicit axes methods.

* Update running scripts.

**Version 2.4.2** Released on 2026-05-16.

* Updated version number in `rename_pscmask_v4_files.sh`.

**Version 2.4.1** Released on 2026-05-16.

* Added a post-processing Bash utility script for automated renaming of output NetCDF files to the standard CALIPSO PSC product naming convention.

**Version 2.4.0** Released on 2026-05-16.

* Harmonized variable names with previous CALIOP product versions.
* Improved NetCDF metadata compliance with CF conventions.
* Computed `Homogeneous_Chunks_Mean_Temperature` and applied a 200 K threshold to distinguish SBS from STS.
* Added `Temperature`, `HNO3_Mixing_Ratio`, and `H2O_Mixing_Ratio` to the NetCDF output files.

**Version 2.3.0** Released on 2026-04-30.

* Averaged NAT/ice thresholds over homogeneous chunks for classification.

**Version 2.2.0** Released on 2026-04-23.

* Added "Likely tropospheric feature" to the classification.

**Version 2.1.0** Released on 2026-04-22.

* Determined liquid/solid phase based on detection in the perpendicular channel.

**Version 2.0.0** Released on 2026-04-21.

* Added a classification function similar to PSCMask V3, applied to homogeneous features.

**Version 1.6.0** Released on 2026-03-31.

* Adjusted algorithm parameters to improve detection of faint PSCs.

**Version 1.5.0** Released on 2026-03-12.

* Filtered low-energy shots.

**Version 1.4.4** Released on 2026-02-04.

* Moved input configuration parameters to a YAML file.

**Version 1.4.3** Released on 2026-02-03.

* Fixed: Avoided `log(0)` by skipping fully invalid meteorological density profiles.
* Added: `global_attrs` to `write_netcdf()`.
* Added: Git-based version identification for development builds.

**Version 1.4.2** Released on 2025-12-11.

* Fixed: Increased the number of values in `FCORR` to avoid errors when `nb_bins_shift > 20`.

**Version 1.4.1** Released on 2025-12-09.

* Fixed: Exit the process if no data are found between `lat_min` and `lat_max`.

**Version 1.4.0** Released on 2025-06-13.

* First version run over the entire CALIPSO period.
