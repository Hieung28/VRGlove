import serial
import struct

PACKET_TYPE_QUAT        = b'\x02'
PACKET_TYPE_YPR         = b'\x03'
PACKET_TYPE_RAW         = b'\x04'
PACKET_TYPE_ACCEL_CALIB = b'\x05'

NUM_IMU = 13
FIX_HEADER_DATA_SIZE = 12
QUAT_SIZE = 4
FLOAT_SIZE = 4
QUAT_PER_FIG = 5

thumb_primary   =  0; thumb_tp  =  1; thumb_mc  =  2; thumb_pp  =  3; thumb_dp  =  4
index_primary   =  5; index_mc  =  6; index_pp  =  7; index_mp  =  8; index_dp  =  9
middle_primary  = 10; middle_mc = 11; middle_pp = 12; middle_mp = 13; middle_dp = 14
ring_secondary  = 15; ring_mc   = 16; ring_pp   = 17; ring_mp   = 18; ring_dp   = 19
pinky_secondary = 20; pinky_mc  = 21; pinky_pp  = 22; pinky_mp  = 23; pinky_dp  = 24
hand_wist = 25

joint_R_forearm_carpal = 0; joint_R_carpal_hand = 1
joint_R_hand_thumb  =  2; joint_R_thumb_tp_mc  =  3; joint_R_thumb_mc_pp  =  4; joint_R_thumb_pp_dp  =  5
joint_R_hand_index  =  6; joint_R_index_mc_pp  =  7; joint_R_index_pp_mp  =  8; joint_R_index_mp_dp  =  9
joint_R_hand_middle = 10; joint_R_middle_mc_pp = 11; joint_R_middle_pp_mp = 12; joint_R_middle_mp_dp = 13
joint_R_hand_ring   = 14; joint_R_ring_mc_pp   = 15; joint_R_ring_pp_mp   = 16; joint_R_ring_mp_dp   = 17
joint_R_hand_pinky  = 18; joint_R_pinky_mc_pp  = 19; joint_R_pinky_pp_mp  = 20; joint_R_pinky_mp_dp  = 21

packet_size_dict = {PACKET_TYPE_QUAT: FIX_HEADER_DATA_SIZE+(1+NUM_IMU)+4*4*NUM_IMU}

def parse_packet(packet_type, packet):
    res = dict()
    if packet_type == PACKET_TYPE_QUAT:
        # first 4 byte is packet counter
        # res.append(struct.unpack('i', packet[:4])[0])
        number_bones = struct.unpack('B', packet[4:5])[0]  # TODO: if NUM_IMU is defined then we don't need this
        bones_with_IMU_index = struct.unpack(str(number_bones)+'B', packet[5:5+number_bones])
        for i in range(number_bones):
            start_index = 5 + number_bones + (i * QUAT_SIZE*FLOAT_SIZE)
            end_index = start_index + QUAT_SIZE * FLOAT_SIZE
            res[bones_with_IMU_index[i]] = struct.unpack('<4f', packet[start_index:end_index])
    return res    

def read_packet(ser):
    while True:
        byte = ser.read()        
        if byte in [PACKET_TYPE_QUAT, PACKET_TYPE_YPR, PACKET_TYPE_RAW, PACKET_TYPE_ACCEL_CALIB]:
            line = ser.read(packet_size_dict[byte]-2)
            if line[-2] == ord('\r') and line[-1] == ord('\n'):
                # print(line)
                res = parse_packet(byte, line[:-2])
                return res            
        return None