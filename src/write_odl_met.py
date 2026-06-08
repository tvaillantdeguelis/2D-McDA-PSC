#!/usr/bin/env python
# coding: utf8

"""
write_odl_met.py

Create an ODL metadata file associated with a PSC netCDF granule.

Example:
    write_odl_met(
        "CAL_LID_L2_PSCMask-Standard-V4-00.2010-01-18T00-19-57ZN.nc"
    )

Creates:
    CAL_LID_L2_PSCMask-Standard-V4-00.2010-01-18T00-19-57ZN.nc.met
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from netCDF4 import Dataset


# ======================================================================
# Helpers
# ======================================================================

def format_odl_object(name, value, indent=4, quoted=True):
    """
    Create an ODL OBJECT block.
    """

    spaces = " " * indent

    if quoted:
        value_str = f"\"{value}\""
    else:
        value_str = str(value)

    return (
        f"{spaces}OBJECT        =  {name}\n"
        f"{spaces}  NUM_VAL     =  1\n"
        f"{spaces}  VALUE       =  {value_str}\n"
        f"{spaces}END_OBJECT    =  {name}"
    )


def tai93_to_datetime(tai_seconds):
    """
    Convert CALIPSO Profile_Time
    (seconds since 1993-01-01)
    into datetime.
    """

    epoch = datetime(1993, 1, 1, tzinfo=timezone.utc)

    return epoch + \
        __import__("datetime").timedelta(seconds=float(tai_seconds))


def get_daynight_flag(latitudes):
    """
    Placeholder.

    Replace by actual day/night computation if desired.
    """

    return "DAY"


def build_gline(latitudes, longitudes, npts=10):
    """
    Build GLINE coordinates using evenly spaced samples.
    """

    idx = np.linspace(
        0,
        len(latitudes) - 1,
        npts,
        dtype=int
    )

    lat = latitudes[idx]
    lon = longitudes[idx]

    lat_txt = ",".join(f"{x:.6f}" for x in lat)
    lon_txt = ",".join(f"{x:.6f}" for x in lon)

    return lat_txt, lon_txt


# ======================================================================
# Main writer
# ======================================================================

def write_odl_met(nc_filename):

    nc_filename = Path(nc_filename)

    met_filename = Path(str(nc_filename) + ".met")

    production_time = (
        datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    with Dataset(nc_filename) as ds:

        profile_time = ds.variables["Profile_Time"][:]

        latitudes = ds.variables["Latitude"][:]
        longitudes = ds.variables["Longitude"][:]

        start_dt = tai93_to_datetime(profile_time[0])
        end_dt   = tai93_to_datetime(profile_time[-1])

        daynight = get_daynight_flag(latitudes)

        version_2d_mcda_psc = ds.getncattr("algorithm_version")

        gline_lat, gline_lon = build_gline(
            latitudes,
            longitudes
        )

    granule_name = nc_filename.name

    shortname = granule_name.split(".")[0]

    version = "V4-00"

    lines = []

    # ==================================================================
    # Header
    # ==================================================================

    lines.append("GROUP       =  INVENTORYMETADATA")
    lines.append("  GROUPTYPE   =  MASTERGROUP")

    # ==================================================================
    # ECSDATAGRANULE
    # ==================================================================

    lines.append("  GROUP       =  ECSDATAGRANULE")

    lines.append(
        format_odl_object(
            "LOCALGRANULEID",
            granule_name,
            quoted=True
        )
    )

    lines.append(
        format_odl_object(
            "LOCALVERSIONID",
            f"2D-McDA-PSC {version_2d_mcda_psc}",
            quoted=True
        )
    )

    lines.append(
        format_odl_object(
            "PRODUCTIONDATETIME",
            production_time,
            quoted=True
        )
    )

    lines.append(
        format_odl_object(
            "DAYNIGHTFLAG",
            daynight,
            quoted=True
        )
    )

    lines.append("  END_GROUP   =  ECSDATAGRANULE")

    # ==================================================================
    # COLLECTIONDESCRIPTIONCLASS
    # ==================================================================

    lines.append("  GROUP       =  COLLECTIONDESCRIPTIONCLASS")

    lines.append(
        format_odl_object(
            "SHORTNAME",
            shortname,
            quoted=True
        )
    )

    lines.append(
        format_odl_object(
            "VERSIONID",
            version,
            quoted=True
        )
    )

    lines.append("  END_GROUP   =  COLLECTIONDESCRIPTIONCLASS")

    # ==================================================================
    # RANGEDATETIME
    # ==================================================================

    lines.append("  GROUP       =  RANGEDATETIME")

    lines.append(
        format_odl_object(
            "RANGEBEGINNINGDATE",
            start_dt.strftime("%Y-%m-%d"),
            quoted=True
        )
    )

    lines.append(
        format_odl_object(
            "RANGEBEGINNINGTIME",
            start_dt.strftime("%H:%M:%SZ"),
            quoted=True
        )
    )

    lines.append(
        format_odl_object(
            "RANGEENDINGDATE",
            end_dt.strftime("%Y-%m-%d"),
            quoted=True
        )
    )

    lines.append(
        format_odl_object(
            "RANGEENDINGTIME",
            end_dt.strftime("%H:%M:%SZ"),
            quoted=True
        )
    )

    lines.append("  END_GROUP   =  RANGEDATETIME")


    # ==================================================================
    # SPATIAL DOMAIN
    # ==================================================================

    lines.append("  GROUP       =  SPATIALDOMAINCONTAINER")
    lines.append("    GROUP       =  HORIZONTALSPATIALDOMAINCONTAINER")
    lines.append("      GROUP       =  GLINE")

    lines.append("        OBJECT        =  GLINELATITUDE")
    lines.append("          CLASS       =  \"1\"")
    lines.append("          NUM_VAL     =  \"10\"")
    lines.append(f"          VALUE       =  ({gline_lat})")
    lines.append("        END_OBJECT    =  GLINELATITUDE")

    lines.append("        OBJECT        =  GLINELONGITUDE")
    lines.append("          CLASS       =  \"1\"")
    lines.append("          NUM_VAL     =  \"10\"")
    lines.append(f"          VALUE       =  ({gline_lon})")
    lines.append("        END_OBJECT    =  GLINELONGITUDE")

    lines.append("      END_GROUP   =  GLINE")
    lines.append("    END_GROUP   =  HORIZONTALSPATIALDOMAINCONTAINER")
    lines.append("  END_GROUP   =  SPATIALDOMAINCONTAINER")

    # ==================================================================
    # Footer
    # ==================================================================

    lines.append("END_GROUP   =  INVENTORYMETADATA")
    lines.append("END")

    with open(met_filename, "w") as f:
        f.write("\n".join(lines))

    print(f"{met_filename} created.")


# ======================================================================
# Command line
# ======================================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage:")
        print("    write_odl_met.py file.nc")
        sys.exit(1)

    write_odl_met(sys.argv[1])