Input Parameter Description


Spot/Band - Whether this row of the file is describing a spot rotating around the body or a band at a specific latitude.

Latitude - The latitude of the spot or the band.

RelBright - The peak brightness of the spot or the brightness amplitude of the band, relative to the background brightness of brightness of the target brown dwarf.

RelVel - The relative rotational velocity of the spot/band relative to the rotational velocity of the target brown dwarf.

Phase - The starting longitude of the spot or the brightest part of the band, with 0.0 starting towards the observer depending on inclination.

MajorDiam - The major diameter of the elliptical spot, in metres.

MinorDiam - The minor diameter of the elliptical spot, in metres.

BandHeight - The height of the band that encircles the brown dwarf, in metres

Permanent - A True/False statement applicaple to spots. If true, then the spot will not be altered in any way throughout the simulation. If false the spot brightness (output) will vary depending on the following two parameters.

SpotBegin - The position on the x-axis that the spot will begin to appear. Note that this is the x-axis value divided by the timestep, i.e. the array element vlaue that will first have a value for the spot brightness.

SpotEnd - Similar to SpotBegin, this marks the time array element that will have the final entry for brightness.
