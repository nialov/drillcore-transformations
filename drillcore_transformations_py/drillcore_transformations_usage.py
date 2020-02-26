import pandas as pd
from pathlib import Path

from drillcore_transformations_py import drillcore_transformations

def round_outputs(number):
	return round(number, 2)

def transform_from_csv(filename, with_gamma=False):
	"""
	Transforms data from a given .csv file. File must have columns:
	['alpha', 'beta', 'borehole_trend', 'borehole_plunge' and 'gamma' if with_gamma == True]
	Saves new .csv file in the same directory with

	:param filename: Path to file for reading.
	:type filename: str
	:param with_gamma: Do gamma calculations or not
	:type with_gamma: bool
	"""
	df = pd.read_csv(filename, sep=';')
	# Creates and calculates new columns
	if with_gamma:
		df[['plane_dip', 'plane_dir', 'gamma_plunge', 'gamma_trend']] = df.apply(
			lambda row: pd.Series(drillcore_transformations.transform_with_gamma(
				row['alpha'], row['beta'], row['borehole_trend'], row['borehole_plunge'], row['gamma'])), axis=1)
		df[['plane_dip', 'plane_dir', 'gamma_plunge', 'gamma_trend']] = df[['plane_dip', 'plane_dir', 'gamma_plunge', 'gamma_trend']].applymap(round_outputs)
	else:
		df[['plane_dip', 'plane_dir']] = df.apply(
			lambda row: pd.Series(drillcore_transformations.transform_without_gamma(
				row['alpha'], row['beta'], row['borehole_trend'], row['borehole_plunge'])), axis=1)
		df[['plane_dip', 'plane_dir']] = df[['plane_dip', 'plane_dir']].applymap(round_outputs)

	# Savename
	savename = Path(filename).stem.split('.')[0] + '_orient_calculated.csv'
	savedir = str(Path(filename).parent)
	# Save new .csv
	df.to_csv(Path(savedir+r'/'+savename), sep=';', mode='w+')

