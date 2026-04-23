**Version 2.2.0** Released 2026-04-23.

* Add "Likely tropospheric feature" to the classification.

**Version 2.1.0** Released 2026-04-22.

* Determine liquid/solid based on detection in perp channel.

**Version 2.0.0** Released 2026-04-21.

* Add classification function. Similar to PSCMask V3 applied to homogeneous features.

**Version 1.6.0** Released 2026-03-31.

* Adjust algorithm parameters to improve detection of faint PSCs.

**Version 1.5.0** Released 2026-03-12.

* Filter low energy shots.

**Version 1.4.4** Released 2026-02-04.

* Move input configuration parameters to YAML file

**Version 1.4.3** Released 2026-02-03.

* Fixed: Avoid log(0) by skipping fully invalid met density profiles
* Added: global_attrs to write_netcdf()
* Added: Git-based version identification for development

**Version 1.4.2** Released 2025-12-11.

* Fixed: Increase number of values in FCORR to avoid error when nb_bins_shift > 20.

**Version 1.4.1** Released 2025-12-09.

* Fixed: Exit process if no data between lat_min and lat_max.

**Version 1.4.0** Released 2025-06-13.

* Fist version run over the whole CALIPSO period.