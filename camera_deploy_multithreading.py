# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2021 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
#  Trigger.py shows how to trigger the camera. It relies on information
#  provided in the Enumeration, Acquisition, and NodeMapInfo examples.
#
#  It can also be helpful to familiarize yourself with the ImageFormatControl
#  and Exposure examples. As they are somewhat shorter and simpler, either
#  provides a strong introduction to camera customization.
#
#  This example shows the process of configuring, using, and cleaning up a
#  camera for use with both a software and a hardware trigger.

import os
import PySpin
import sys
import atexit
import argparse
import time
import numpy as np
import threading
run_event = threading.Event()
#NUM_IMAGES = 10  # number of images to grab


class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2


CHOSEN_TRIGGER = TriggerType.SOFTWARE
AUTO_EXPOSURE = True

trigger_delay_to_set = 500.
exposure_time_to_set = 750.0
gain_to_set = 5.
wb_red = 1.34
wb_blue = 2.98
second_per_frame = 0.45 #0.33#1.25


# =======================  real time processing ======================= 

trt_fp16Path = '/media/nvidia/Data/trtModel/fp16/newnet_resnet18_64_newHead_m2_a1.trt'
severity_output_dir = 'None'
mask_output_dir = 'None'
dataPath = 'None'


from Xavier_SpeedTest_folder_function import imageProcessor
from ImageMask2Tiles_uniform import ImageMask2Tiles
imgProcessor = None 

image_converted_queue = []
image_status_queue = []
queue_modifier_lock = threading.Lock()

def inference(input_img,imname):
    global mask_output_dir
    global imgProcessor
    global severity_output_dir
    isOutput = mask_output_dir != 'None'
    infectedArea,leaveArea,assembledImg = imgProcessor.inference(input_img,dump=isOutput)
    with open (severity_output_dir + '/' + imname + '.txt', 'w') as f:
        f.write(str(infectedArea)+','+str(leaveArea))
    if isOutput:
        imname = imname+'.png'
        cv2.imwrite(mask_output_dir+'/'+imname,assembledImg,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        
def inference_multiThreading():        
    global image_converted_queue
    global image_status_queue
    global run_event
    global queue_modifier_lock
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
    starttime4 = -1
    imname,isProcessed,isSaved = ['',0,0]
    
    isHandled = 1
    while run_event.is_set():
        #try:
        if True:
            queue_modifier_lock.acquire()
            if len(image_converted_queue) != 0:
                image_converted = image_converted_queue[0]
                imname,isProcessed,isSaved = image_status_queue[0]
                isHandled = 0
            queue_modifier_lock.release()
            if imname == '':
                continue
            #print(imname,isProcessed,isSaved)
            if isHandled == 0:
                isHandled = 1
                if isProcessed == 0:
                    if starttime4 != -1:
                        print("!!!!!!!!!!!idle time: %s" % (time.time()-starttime4))
                        starttime4 = -1
                    starttime3 = time.time()
                    inference(image_converted,imname)
                    print('Inference Time: %s' % (time.time()-starttime3))
                    image_status_queue[0][1] = 1               
            else:
                if starttime4 == -1:
                    starttime4 = time.time()
        #except Exception as e:
        #    print(e)
        #    break
        

def saving_multiThreading():
    global image_converted_queue
    global image_status_queue
    global run_event
    global dataPath
    global queue_modifier_lock
    print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYyyy")
    starttime4 = -1
    imname,isProcessed,isSaved = ['',0,0]
    isHandled = 1
    while run_event.is_set():
        if True:
        #try:
            queue_modifier_lock.acquire()
            if len(image_converted_queue) != 0:
                image_converted = image_converted_queue[0]
                imname,isProcessed,isSaved = image_status_queue[0]
                isHandled = 0
            queue_modifier_lock.release()
            if imname == '':
                continue
            #print(imname,isProcessed,isSaved)
            if isHandled == 0:
                isHandled = 1
                if isSaved == 0:
                    if starttime4 != -1:
                        print("!!!!!!!!!!!idle time: %s" % (time.time()-starttime4))
                        starttime4 = -1
                    starttime3 = time.time()
                    with open(dataPath+'/'+imname+'.npy', 'wb') as f:
                        np.save(f,image_converted)
                        print('Image saved at %s' % imname)
                    print('Saving Time: %s' % (time.time()-starttime3))
                    
                    image_status_queue[0][2] = 1        
            else:
                if starttime4 == -1:
                    starttime4 = time.time()
        #except Exception as e:
        #    print(e)
        #    break
        
def manageQueue():
    global image_converted_queue
    global image_status_queue
    global run_event
    global dataPath
    global queue_modifier_lock
    print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZz")
    starttime4 = -1
    imname,isProcessed,isSaved = ['',0,0]
    while run_event.is_set():
        time.sleep(0.1)
        if True:
        #try:
            queue_modifier_lock.acquire()
            if len(image_converted_queue) != 0:
                imname,isProcessed,isSaved = image_status_queue[0]
                if isProcessed == 1 and isSaved == 1:
                    image_converted_queue.pop(0)
                    image_status_queue.pop(0)        
            queue_modifier_lock.release()
            

# ======================= end real time processing ======================= 
           

def configure_Packet(cam):
    """
     This function configures a custom exposure time. Automatic exposure is turned
     off in order to allow for the customization, and then the custom setting is
     applied.

     :param cam: Camera to configure exposure for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """
    max_packet = 9000        
    packet_delay = 10
    result = True
    print('*** CONFIGURING PACKET ***\n')
    try:
        if cam.GevSCPSPacketSize.GetAccessMode() != PySpin.RW:
            print('Unable to set Packet Size. Aborting...')
            return True

        if cam.GevSCPD.GetAccessMode() != PySpin.RW:
            print('Unable to set Packet delay. Aborting...')
            return True
            
            
            
        

        # Ensure desired exposure time does not exceed the maximum
        #exposure_time_to_set = 200.0

        cam.GevSCPSPacketSize.SetValue(max_packet)
        cam.GevSCPD.SetValue(packet_delay)
        print('Packet size set to %s...\n' % max_packet)
        print('Packet delay set to %s...\n' % packet_delay)
    except Exception as ex:
        print('Error: %s' % ex)
        result = False

    return result




def configure_wb(cam):
    global wb_red
    global wb_blue
    print('*** CONFIGURING white balance ***\n')

    try:
        result = True
        nodemap = cam.GetNodeMap()
        node_wb = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
        if not PySpin.IsAvailable(node_wb) or not PySpin.IsReadable(node_wb):
            print('Unable to disable auto wb (node retrieval). Aborting...')
            return False

        node_wb_off = node_wb.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_wb_off) or not PySpin.IsReadable(node_wb_off):
            print('Unable to disable auto wb (enum entry retrieval). Aborting...')
            return False

        node_wb.SetIntValue(node_wb_off.GetValue())
        print('Automatic WB disabled...')

        node_wb_selector = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceRatioSelector'))
        print(PySpin.IsWritable(node_wb_selector))
        node_wb_red = node_wb_selector.GetEntryByName('Red')
        node_wb_selector.SetIntValue(node_wb_red.GetValue())
        node_wb_ratio = PySpin.CFloatPtr(nodemap.GetNode('BalanceRatio'))
        node_wb_ratio.SetValue(wb_red)

        node_wb_blue = node_wb_selector.GetEntryByName('Blue')
        node_wb_selector.SetIntValue(node_wb_blue.GetValue())
        node_wb_ratio = PySpin.CFloatPtr(nodemap.GetNode('BalanceRatio'))
        node_wb_ratio.SetValue(wb_blue)
        


        print('WB set')

    except Exception as ex:
        print('Error: %s' % ex)
        result = False
    return result

def reset_wb(cam):
    """
    This function returns the camera to a normal state by re-enabling automatic exposure.

    :param cam: Camera to reset exposure on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        nodemap = cam.GetNodeMap()
        node_wb = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceWhiteAuto'))
        if not PySpin.IsAvailable(node_wb) or not PySpin.IsReadable(node_wb):
            print('Unable to enable auto wb (node retrieval). Aborting...')
            return False

        node_wb_continuous = node_wb.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_wb_continuous) or not PySpin.IsReadable(node_wb_continuous):
            print('Unable to enable auto wb (enum entry retrieval). Aborting...')
            return False

        node_wb.SetIntValue(node_wb_continuous.GetValue())
        print('Automatic WB enabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result





def configure_gain(cam):
    """
     This function configures a custom exposure time. Automatic exposure is turned
     off in order to allow for the customization, and then the custom setting is
     applied.

     :param cam: Camera to configure exposure for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """
    global gain_to_set
    print('*** CONFIGURING EXPOSURE ***\n')

    try:
        result = True

        # Turn off automatic exposure mode
        #
        # *** NOTES ***
        # Automatic exposure prevents the manual configuration of exposure
        # times and needs to be turned off for this example. Enumerations
        # representing entry nodes have been added to QuickSpin. This allows
        # for the much easier setting of enumeration nodes to new values.
        #
        # The naming convention of QuickSpin enums is the name of the
        # enumeration node followed by an underscore and the symbolic of
        # the entry node. Selecting "Off" on the "ExposureAuto" node is
        # thus named "ExposureAuto_Off".
        #
        # *** LATER ***
        # Exposure time can be set automatically or manually as needed. This
        # example turns automatic exposure off to set it manually and back
        # on to return the camera to its default state.

        if cam.GainAuto.GetAccessMode() != PySpin.RW:
            print('Unable to disable automatic gain. Aborting...')
            return False

        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        print('Automatic gain disabled...')

        # Set exposure time manually; exposure time recorded in microseconds
        #
        # *** NOTES ***
        # Notice that the node is checked for availability and writability
        # prior to the setting of the node. In QuickSpin, availability and
        # writability are ensured by checking the access mode.
        #
        # Further, it is ensured that the desired exposure time does not exceed
        # the maximum. Exposure time is counted in microseconds - this can be
        # found out either by retrieving the unit with the GetUnit() method or
        # by checking SpinView.

        if cam.Gain.GetAccessMode() != PySpin.RW:
            print('Unable to set gain. Aborting...')
            return False

        # Ensure desired exposure time does not exceed the maximum
        #exposure_time_to_set = 200.0
        gain_to_set = min(cam.Gain.GetMax(), gain_to_set)
        cam.Gain.SetValue(gain_to_set)
        print('Gain set to %s us...\n' % gain_to_set)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def reset_gain(cam):
    """
    This function returns the camera to a normal state by re-enabling automatic exposure.

    :param cam: Camera to reset exposure on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Turn automatic exposure back on
        #
        # *** NOTES ***
        # Automatic exposure is turned on in order to return the camera to its
        # default state.

        if cam.GainAuto.GetAccessMode() != PySpin.RW:
            print('Unable to enable automatic gain (node retrieval). Non-fatal error...')
            return False

        cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)

        print('Automatic gain enabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def configure_exposure(cam):
    """
     This function configures a custom exposure time. Automatic exposure is turned
     off in order to allow for the customization, and then the custom setting is
     applied.

     :param cam: Camera to configure exposure for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """
    global exposure_time_to_set
    print('*** CONFIGURING EXPOSURE ***\n')

    try:
        result = True

        # Turn off automatic exposure mode
        #
        # *** NOTES ***
        # Automatic exposure prevents the manual configuration of exposure
        # times and needs to be turned off for this example. Enumerations
        # representing entry nodes have been added to QuickSpin. This allows
        # for the much easier setting of enumeration nodes to new values.
        #
        # The naming convention of QuickSpin enums is the name of the
        # enumeration node followed by an underscore and the symbolic of
        # the entry node. Selecting "Off" on the "ExposureAuto" node is
        # thus named "ExposureAuto_Off".
        #
        # *** LATER ***
        # Exposure time can be set automatically or manually as needed. This
        # example turns automatic exposure off to set it manually and back
        # on to return the camera to its default state.

        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to disable automatic exposure. Aborting...')
            return False

        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        print('Automatic exposure disabled...')

        # Set exposure time manually; exposure time recorded in microseconds
        #
        # *** NOTES ***
        # Notice that the node is checked for availability and writability
        # prior to the setting of the node. In QuickSpin, availability and
        # writability are ensured by checking the access mode.
        #
        # Further, it is ensured that the desired exposure time does not exceed
        # the maximum. Exposure time is counted in microseconds - this can be
        # found out either by retrieving the unit with the GetUnit() method or
        # by checking SpinView.

        if cam.ExposureTime.GetAccessMode() != PySpin.RW:
            print('Unable to set exposure time. Aborting...')
            return False

        # Ensure desired exposure time does not exceed the maximum
        #exposure_time_to_set = 200.0
        exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
        cam.ExposureTime.SetValue(exposure_time_to_set)
        print('Shutter time set to %s us...\n' % exposure_time_to_set)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def reset_exposure(cam):
    """
    This function returns the camera to a normal state by re-enabling automatic exposure.

    :param cam: Camera to reset exposure on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Turn automatic exposure back on
        #
        # *** NOTES ***
        # Automatic exposure is turned on in order to return the camera to its
        # default state.

        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
            return False

        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)

        print('Automatic exposure enabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def configure_trigger(cam):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    set to off in order to select the trigger source. Once the trigger source
    has been selected, trigger mode is then enabled, which has the camera
    capture only a single image upon the execution of the chosen trigger.

     :param cam: Camera to configure trigger for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """

    global trigger_delay_to_set
    result = True

    print('*** CONFIGURING TRIGGER ***\n')

    print('Note that if the application / user software triggers faster than frame time, the trigger may be dropped / skipped by the camera.\n')
    print('If several frames are needed per trigger, a more reliable alternative for such case, is to use the multi-frame mode.\n\n')

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
        print('Software trigger chosen ...')
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
        print('Hardware trigger chose ...')

    try:
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        nodemap = cam.GetNodeMap()
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')
        
        # Set TriggerSelector to FrameStart
        # For this example, the trigger selector should be set to frame start.
        # This is the default for most cameras.
        node_trigger_selector= PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))

        if not PySpin.IsAvailable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
            print('Unable to get trigger selector (node retrieval). Aborting...')
            return False

        node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
        if not PySpin.IsAvailable(node_trigger_selector_framestart) or not PySpin.IsReadable(
                node_trigger_selector_framestart):
            print('Unable to set trigger selector (enum entry retrieval). Aborting...')
            return False
        node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())
        
        print('Trigger selector set to frame start...')

        cam.TriggerDelay.SetValue(trigger_delay_to_set)
        print('Trigger delay set')

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
        # mode is off.
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        
        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
            if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                    node_trigger_source_software):
                print('Unable to set trigger source (enum entry retrieval). Aborting...')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
            print('Trigger source set to software...')

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')

            if not PySpin.IsAvailable(node_trigger_source_hardware) or not PySpin.IsReadable(
                    node_trigger_source_hardware):
                print('Unable to set trigger source (enum entry retrieval). Aborting...')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())
            
            node_trigger_activation = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerActivation'))
            if not PySpin.IsAvailable(node_trigger_activation) or not PySpin.IsWritable(node_trigger_activation):
                print('Unable to get trigger activation (node retrieval). Aborting...')
                return False
            node_trigger_activation_fallingedge = node_trigger_activation.GetEntryByName('FallingEdge')

            if not PySpin.IsAvailable(node_trigger_activation_fallingedge) or not PySpin.IsReadable(node_trigger_activation_fallingedge):
                print('Unable to set trigger activation (enum entry retrieval). Aborting...')
                return False
            node_trigger_activation.SetIntValue(node_trigger_activation_fallingedge.GetValue())

            print('Trigger source set to hardware...')

        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
        if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
            print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
        print('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result



def reset_trigger(nodemap):
    """
    This function returns the camera to a normal state by turning off trigger mode.
  
    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def grab_next_image_by_trigger(cam_list,ser):
    """
    This function acquires an image by executing the trigger node.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Use trigger to capture image
        # The software trigger only feigns being executed by the Enter key;
        # what might not be immediately apparent is that there is not a
        # continuous stream of images being captured; in other examples that
        # acquire images, the camera captures a continuous stream of images.
        # When an image is retrieved, it is plucked from the stream.

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            # Get user input
            #input('Press the Enter key to initiate software trigger.')

            # Execute software trigger
            for cam in cam_list:
                nodemap = cam.GetNodeMap()
                node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
                if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
                    print('Unable to execute trigger. Aborting...')
                    return False,0

                node_softwaretrigger_cmd.Execute()

            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            print('Use the hardware to trigger image acquisition.')
            startTrigger(ser)
        triggerTime = time.time()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False,0

    return result,triggerTime


def initializeTrigger():
    pass
    

def startTrigger(ser):
    print('start trigger...')
    ser.write(b'B1\n')
    
    
def stopTrigger(ser):
    print('stop trigger...')
    ser.write(b'B0\n')

def initializeStatusOutput():
    print('initialize status output')
    

def startStatusOutput(ser):
    print("sending R1...")
    ser.write(b'R1\n')
    
def stopStatusOutput(ser):
    print("sending R0...")
    ser.write(b'R0\n')



def acquire_images(cam_list,dataPath,ser):
    """
    This function acquires and saves 10 images from each device.

    :param cam_list: List of cameras
    :type cam_list: CameraList
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    imgOption = PySpin.PNGOption()
    imgOption.compressionLevel=0
    global second_per_frame
    global image_converted_queue
    global image_status_queue
    global run_event
    global queue_modifier_lock
  

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Prepare each camera to acquire images
        #
        # *** NOTES ***
        # For pseudo-simultaneous streaming, each camera is prepared as if it
        # were just one, but in a loop. Notice that cameras are selected with
        # an index. We demonstrate pseduo-simultaneous streaming because true
        # simultaneous streaming would require multiple process or threads,
        # which is too complex for an example.
        #

        for i, cam in enumerate(cam_list):

            # Set acquisition mode to continuous
            node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n' % i)
                return False

            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
                Aborting... \n' % i)
                return False

            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Camera %d acquisition mode set to continuous...' % i)

            # Begin acquiring images
            cam.BeginAcquisition()

            print('Camera %d started acquiring images...' % i)

            print()

        # Retrieve, convert, and save images for each camera
        #
        # *** NOTES ***
        # In order to work with simultaneous camera streams, nested loops are
        # needed. It is important that the inner loop be the one iterating
        # through the cameras; otherwise, all images will be grabbed from a
        # single camera before grabbing any images from another.
        #for n in range(NUM_IMAGES):
        n = 0
        numBroken = 0
        overTimeCounter = 0
        starttime = time.time()
        while run_event.is_set():
            starttime2 = time.time()
            try:
                #if imname != '':
                #    overTimeCounter += 1
                result,triggerTime = grab_next_image_by_trigger(cam_list,ser)
                if CHOSEN_TRIGGER == TriggerType.HARDWARE:
                    time.sleep(0.001)
                    stopTrigger(ser)
                for i, cam in enumerate(cam_list):
                    if True: #try:
                        # Retrieve device serial number for filename
                        node_device_serial_number = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))

                        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                            device_serial_number = node_device_serial_number.GetValue()
                            print('Camera %d serial number set to %s...' % (i, device_serial_number))

                        # Retrieve next received image and ensure image completion.
                        getImgTimer = time.time()
                        image_result = cam.GetNextImage(5000)
                        print("get image took: ", time.time()-getImgTimer)
                        if image_result.IsIncomplete():
                            print('Image incomplete with image status %d ... \n' % image_result.GetImageStatus())
                            numBroken = numBroken + 1
                        else:
                            received_time = triggerTime*10**6 #int(time.time()*10**6)
                            # Print image information
                            width = image_result.GetWidth()
                            height = image_result.GetHeight()
                            print('Camera %d grabbed image %d, width = %d, height = %d' % (i, n, width, height))
   
                            # Convert image to mono 8
                            image_converted = image_result.Convert(PySpin.PixelFormat_RGB8)#, PySpin.HQ_LINEAR)
                            image_converted = image_converted.GetData().reshape(height, width, 3).copy()

                            # Create a unique filename
                            if device_serial_number:
                                #filename = dataPath + 'AcquisitionMultipleCamera-%s-%d.jpg' % (device_serial_number, n)
                                filename = dataPath + '%s-%d.npy' % (device_serial_number, received_time) 
                            else:
                                #filename = dataPath + 'AcquisitionMultipleCamera-%d-%d.jpg' % (i, n)
                                filename = dataPath + 'AcquisitionMultipleCamera-%d-%d.npy' % (i, received_time)
                     
                            
                            # Save image
                            #image_converted.Save(filename)
                            
                            #with open(filename, 'wb') as f:
                            #    np.save(f,image_converted)
                            #print('Image saved at %s' % filename)
                            imname = '%s-%d' % (device_serial_number, received_time)
                            queue_modifier_lock.acquire()
                            image_converted_queue.append(image_converted)
                            image_status_queue.append([imname,0,0])
                            queue_modifier_lock.release()
                        # Release image
                        image_result.Release()
                        print()

                    #except Exception as ex:
                    #    print('Error: %s' % ex)
                    #    result = False
                n = n + 1
                print('Time: %s' % (time.time()-starttime2))
                print('Broken images: %s' % numBroken)
                print('Overtime: %s' % overTimeCounter)
                print('to be processed: ', image_status_queue)
                time.sleep(second_per_frame-((time.time()-starttime)%second_per_frame))
                
            except Exception as e:
                print(e)

        # End acquisition for each camera
        #
        # *** NOTES ***
        # Notice that what is usually a one-step process is now two steps
        # because of the additional step of selecting the camera. It is worth
        # repeating that camera selection needs to be done once per loop.
        #
        # It is possible to interact with cameras through the camera list with
        # GetByIndex(); this is an alternative to retrieving cameras as
        # CameraPtr objects that can be quick and easy for small tasks.
        for cam in cam_list:

            # End acquisition
            cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result
    





def print_device_info(nodemap, cam_num):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :param cam_num: Camera number.
    :type nodemap: INodeMap
    :type cam_num: int
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('Printing device information for camera %d... \n' % cam_num)

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')
        print()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result




def run_multiple_cameras(cam_list,dataPath,ser):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam_list: List of cameras
    :type cam_list: CameraList
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Retrieve transport layer nodemaps and print device information for
        # each camera
        # *** NOTES ***
        # This example retrieves information from the transport layer nodemap
        # twice: once to print device information and once to grab the device
        # serial number. Rather than caching the nodem#ap, each nodemap is
        # retrieved both times as needed.
        print('*** DEVICE INFORMATION ***\n')

        for i, cam in enumerate(cam_list):

            # Retrieve TL device nodemap
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Print device information
            result &= print_device_info(nodemap_tldevice, i)

        # Initialize each camera
        #
        # *** NOTES ***
        # You may notice that the steps in this function have more loops with
        # less steps per loop; this contrasts the AcquireImages() function
        # which has less loops but more steps per loop. This is done for
        # demonstrative purposes as both work equally well.
        #
        # *** LATER ***
        # Each camera needs to be deinitialized once all images have been
        # acquired.
        for i, cam in enumerate(cam_list):

            # Initialize camera
            cam.Init()

            # Retrieve GenICam nodemap
            #nodemap = cam.GetNodeMap()


            #CHECK
            #if configure_Packet(cam) is False:
            #    return False
                
            # Configure trigger
            if configure_trigger(cam) is False:
                return False
            if AUTO_EXPOSURE:
                if not (reset_exposure(cam) and reset_gain(cam) and reset_wb(cam)):
                    return False
            else:
                # Configure exposure  
                if not (configure_exposure(cam) and configure_gain(cam) and configure_wb(cam)):
                    return False


        # Acquire images on all cameras
        
        
        result &= acquire_images(cam_list,dataPath,ser)

        # Deinitialize each camera
        #
        # *** NOTES ***
        # Again, each camera must be deinitialized separately by first
        # selecting the camera and then deinitializing it.
        for cam in cam_list:

            # Deinitialize camera

            # Reset trigger
            nodemap = cam.GetNodeMap()
            result &= reset_trigger(nodemap)

            # Reset exposure
            result &= reset_exposure(cam)
            result &= reset_gain(cam)
            result &= reset_wb(cam)
            cam.DeInit()

            # Release reference to camera
            # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
            # cleaned up when going out of scope.
            # The usage of del is preferred to assigning the variable to None.
            del cam

    except Exception as ex:
        print('Error: %s' % ex)
        result = False

    return result






def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Since this application saves images in the current folder
    # we must ensure that we have permission to write to this folder.
    # If we do not have permission, fail right away.
    global CHOSEN_TRIGGER
    global AUTO_EXPOSURE
    parser = argparse.ArgumentParser(description='get output dir')
    parser.add_argument('output_dir',type = str)
    parser.add_argument('trigger',type = int)
    parser.add_argument('auto_exposure',type = int)
    args = parser.parse_args()
    
    if args.trigger == 0:
        CHOSEN_TRIGGER = TriggerType.HARDWARE
        import serial
    else:
        CHOSEN_TRIGGER = TriggerType.SOFTWARE
        ser = None
    AUTO_EXPOSURE = args.auto_exposure == 1

    #initialize
    def single_yes_or_no_question(question, default_no=True):
        choices = ' [y/N]: ' if default_no else ' [Y/n]: '
        default_answer = 'n' if default_no else 'y'
        print(str(question+choices))
        reply = str(input()).lower().strip() or default_answer
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False
        else:
            return False if default_no else True

    #initializeTrigger()
    #initializeStatusOutput()

    if CHOSEN_TRIGGER == TriggerType.HARDWARE:
        ser = serial.Serial('/dev/ttyACM0', 9600)  #serial
        ser.close()
        ser.open()

        startStatusOutput(ser)
        question =  "Check wire connections. Ready to continue?"
        answer = single_yes_or_no_question(question, default_no=True)
        if answer == False:
            return False

        startStatusOutput(ser)

        question =  "Check Arduino LED is on. Ready to continue?"
        answer = single_yes_or_no_question(question, default_no=True)
        if answer == False:
            return False

        stopStatusOutput(ser)


        question =  "Check Arduino LED is off. Turn on 72V power supply. Ready to continue?"
        answer = single_yes_or_no_question(question, default_no=True)
        if answer == False:
            return False

        startStatusOutput(ser)



    global dataPath
    dataPath = args.output_dir+'/'
    global severity_output_dir
    severity_output_dir = args.output_dir+'_severity/'

    if os.path.exists(dataPath):
        print('folder exists')
        #return False
    else:
        os.mkdir(dataPath)
    
    if os.path.exists(severity_output_dir):
        print('folder exists')
        #return False
    else:
        os.mkdir(severity_output_dir)

    try:
        test_file = open(dataPath+'test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    global imgProcessor
    ImageMask2Tiles_instance = ImageMask2Tiles((3000,4096,3),(1500,2048,3),transfer_cuda = True)
    imgProcessor = imageProcessor('ignored/newnet_resnet18_64_newHead_m2_a1.pth','reference.png',ImageMask2Tiles_instance)
    #imgProcessor = imageProcessor('ignored/AttanetRN18.pth','reference.png',ImageMask2Tiles_instance)
    #if args.trigger == 0:
    #    startTrigger()
    # Run example on each camera
    print("check")
    run_event.set()
    t1 = threading.Thread(target=run_multiple_cameras, args=(cam_list,dataPath,ser,))
    t2 = threading.Thread(target=inference_multiThreading)
    t3 = threading.Thread(target=saving_multiThreading)
    t4 = threading.Thread(target=manageQueue)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    try:
        while 1:
            time.sleep(.1)
    except KeyboardInterrupt:
        run_event.clear()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        print("all threads closed!")
    #result = run_multiple_cameras(cam_list,dataPath,ser)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.

    if args.trigger == 0:
        stopTrigger(ser)
        stopStatusOutput(ser)
    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()


    input('Done! Press Enter to exit...')
    return result


def exit_handler():

    if CHOSEN_TRIGGER == TriggerType.HARDWARE:
        import serial
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout = 0)  #serial
        ser.close()
        ser.open()
        stopTrigger(ser)
        stopStatusOutput(ser)

if __name__ == '__main__':
    atexit.register(exit_handler)
    try:
        if main():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)
