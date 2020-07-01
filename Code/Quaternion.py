from math import *

def getProduct(q1, q2):
	w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
	x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
	y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
	z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
	return w, x, y, z

def getConjugate(q):
	w =  q[0]
	x = -q[1]
	y = -q[2]
	z = -q[3]
	return w, x, y, z

def getInverse(q):
	q = getConjugate(getNormalized(q))
	return q

def getMagnitude(q):
	m = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
	return m

def getNormalized(q):
	m = getMagnitude(q)
	if m != 0:
		w = q[0]/m
		x = q[1]/m
		y = q[2]/m
		z = q[3]/m
		return w, x, y, z
	return q

def getEulerAngle(q):
	q = getNormalized(q)

	# roll (x-axis rotation)
	sinr_cosp = +2.0*(q[0]*q[1] + q[2]*q[3])
	cosr_cosp = +1.0 - 2.0*(q[1]*q[1] + q[2]*q[2])
	roll = atan2(sinr_cosp, cosr_cosp)

	# pitch (y-axis rotation)
	sinp = +2.0*(q[0]*q[2] - q[3]*q[1])
	if fabs(sinp) >= 1:
		pitch = copysign(pi/2, sinp) # use 90 degrees if out of range
	else:
		pitch = asin(sinp)

	# yaw (z-axis rotation)
	siny_cosp = +2.0*(q[0]*q[3] + q[1]*q[2])
	cosy_cosp = +1.0 - 2.0*(q[2]*q[2] + q[3]*q[3])
	yaw = atan2(siny_cosp, cosy_cosp)

	return roll, pitch, yaw

def getDegree(q):
	roll, pitch, yaw = getEulerAngle(q)
	roll  = degrees(roll)
	pitch = degrees(pitch)
	yaw   = degrees(yaw)
	return roll, pitch, yaw

def getRelativeAngle(q1, q2):
	roll, pitch, yaw = getEulerAngle(getProduct(getInverse(q1), q2))
	return roll, pitch, yaw

def rotateYAxis(q):
	q2 = (0, 0, -1, 0)
	q = getProduct(q2, q)
	return q