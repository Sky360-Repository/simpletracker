import os
import time
import numpy as np
from functools import partial
# import matplotlib.pyplot as plt
import cv2 as cv

def CheckFg(fg_gs,fg):
    if (len(fg_gs) != len(fg)):
        print(f'len(fg_gs) == {len(fg_gs)} and len(fg) == {len(fg)}')
        return False
    for i in range(0,len(fg)):
        if(np.sum(fg_gs[i]!=fg[i]) != 0):
            print(f'fg_gs[{i}] != fg[{i}]')
            return i
    print('Test passed!')
    return True

#export
# globals
vidPath = './vids/birds_and_plane.mp4'
lr = 0.05
rows_big = 1440
cols_big = 2560
check_res = False
frame_device = cv.cuda_GpuMat()

#export
def ProcVid0(proc_frame_func,lr):
    cap = cv.VideoCapture(vidPath)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        return
    n_frames = 0
    start = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            n_frames += 1
            proc_frame_func(frame,lr)
        else:
            break
    end = time.time()
    cap.release()
    return (end - start)*1000/n_frames, n_frames;

#export
bgmog2 = cv.createBackgroundSubtractorMOG2()
def ProcFrameCPU0(frame,lr,store_res=False):
    frame_big = cv.resize(frame,(cols_big,rows_big))
    fg_big = bgmog2.apply(frame_big,learningRate = lr)
    fg_small = cv.resize(fg_big,(frame.shape[1],frame.shape[0]))
    if(store_res):
        cpu_res.append(np.copy(fg_small))

# export
cpu_res = []
cpu_time_0, n_frames = ProcVid0(partial(ProcFrameCPU0, store_res=check_res), lr)
print(f'CPU 0 (naive): {n_frames} frames, {cpu_time_0:.2f} ms/frame')

#export
bgmog2_device = cv.cuda.createBackgroundSubtractorMOG2()
def ProcFrameCuda0(frame,lr,store_res=False):
    frame_device.upload(frame)
    frame_device_big = cv.cuda.resize(frame_device,(cols_big,rows_big))
    fg_device_big = bgmog2_device.apply(frame_device_big,lr,cv.cuda.Stream_Null())
    fg_device = cv.cuda.resize(fg_device_big,frame_device.size())
    fg_host = fg_device.download()
    if(store_res):
        gpu_res.append(np.copy(fg_host))

#export
gpu_res = []
gpu_time_0, n_frames = ProcVid0(partial(ProcFrameCuda0,store_res=check_res),lr)
print(f'GPU 0 (naive): {n_frames} frames, {gpu_time_0:.2f} ms/frame')
print(f'Speedup over CPU: {cpu_time_0/gpu_time_0:.2f}')

#export
def ProcVid1(proc_frame,lr):
    cap = cv.VideoCapture(vidPath)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        return
    n_frames = 0
    start = time.time()
    while(cap.isOpened()):
        ret,_ = cap.read(proc_frame.Frame())
        if ret == True:
            n_frames += 1
            proc_frame.ProcessFrame(lr)
        else:
            break
    end = time.time()
    cap.release()
    return (end - start)*1000/n_frames, n_frames;

# export
class ProcFrameCpu1:
    def __init__(self, rows_small, cols_small, rows_big, cols_big, store_res=False):
        self.rows_small, self.cols_small, self.rows_big, self.cols_big = rows_small, cols_small, rows_big, cols_big
        self.store_res = store_res
        self.res = []
        self.bgmog2 = cv.createBackgroundSubtractorMOG2()
        self.frame = np.empty((rows_small, cols_small, 3), np.uint8)
        self.frame_big = np.empty((rows_big, cols_big, 3), np.uint8)
        self.fg_big = np.empty((rows_big, cols_big), np.uint8)
        self.fg_small = np.empty((rows_small, cols_small), np.uint8)

    def ProcessFrame(self, lr):
        cv.resize(self.frame, (self.cols_big, self.rows_big), self.frame_big)
        self.bgmog2.apply(self.frame_big, self.fg_big, learningRate=lr)
        cv.resize(self.fg_big, (self.cols_small, self.rows_small), self.fg_small)
        if (self.store_res):
            self.res.append(np.copy(self.fg_small))

    def Frame(self):
        return self.frame

cap = cv.VideoCapture(vidPath)
if (cap.isOpened() == False):
    print("Error opening video stream or file")
ret, frame = cap.read()
cap.release()
rows_small, cols_small = frame.shape[:2]
proc_frame_cpu1 = ProcFrameCpu1(rows_small, cols_small, rows_big, cols_big, check_res)

#export
cpu_time_1, n_frames = ProcVid1(proc_frame_cpu1,lr)
print(f'CPU 1 (pre-allocation): {n_frames} frames, {cpu_time_1:.2f} ms/frame')
print(f'Speedup over CPU baseline: {cpu_time_0/cpu_time_1:.2f}')

if check_res: CheckFg(cpu_res,proc_frame_cpu1.res)

# export
class ProcFrameCuda1:
    def __init__(self, rows_small, cols_small, rows_big, cols_big, store_res=False):
        self.rows_small, self.cols_small, self.rows_big, self.cols_big = rows_small, cols_small, rows_big, cols_big
        self.store_res = store_res
        self.res = []
        self.bgmog2 = cv.cuda.createBackgroundSubtractorMOG2()
        self.frame = np.empty((rows_small, cols_small, 3), np.uint8)
        self.frame_device = cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC3)
        self.frame_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC3)
        self.fg_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC1)
        self.fg_device_big.setTo(0)
        self.fg_device = cv.cuda_GpuMat(np.shape(frame)[0], np.shape(frame)[1], cv.CV_8UC1)
        self.fg_host = np.empty((rows_small, cols_small), np.uint8)

    def ProcessFrame(self, lr):
        self.frame_device.upload(self.frame)
        cv.cuda.resize(self.frame_device, (cols_big, rows_big), self.frame_device_big)
        self.bgmog2.apply(self.frame_device_big, lr, cv.cuda.Stream_Null(), self.fg_device_big)
        cv.cuda.resize(self.fg_device_big, self.fg_device.size(), self.fg_device)
        self.fg_device.download(self.fg_host)
        if (self.store_res):
            self.res.append(np.copy(self.fg_host))

    def Frame(self):
        return self.frame

proc_frame_cuda1 = ProcFrameCuda1(rows_small, cols_small, rows_big, cols_big, check_res)

#export
gpu_time_1, n_frames = ProcVid1(proc_frame_cuda1,lr)
print(f'GPU 1 (pre-allocation): {n_frames} frames, {gpu_time_1:.2f} ms/frame')
print(f'Incremental speedup: {gpu_time_0/gpu_time_1:.2f}')
print(f'Speedup over CPU: {cpu_time_1/gpu_time_1:.2f}')

if check_res: CheckFg(gpu_res,proc_frame_cuda1.res)

# export
class ProcFrameCuda2:
    def __init__(self, rows_small, cols_small, rows_big, cols_big, store_res=False):
        self.rows_small, self.cols_small, self.rows_big, self.cols_big = rows_small, cols_small, rows_big, cols_big
        self.store_res = store_res
        self.res = []
        self.bgmog2 = cv.cuda.createBackgroundSubtractorMOG2()
        self.stream = cv.cuda_Stream()
        self.frame = np.empty((rows_small, cols_small, 3), np.uint8)
        self.frame_device = cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC3)
        self.frame_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC3)
        self.fg_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC1)
        self.fg_device = cv.cuda_GpuMat(np.shape(frame)[0], np.shape(frame)[1], cv.CV_8UC1)
        self.fg_host = np.empty((rows_small, cols_small), np.uint8)

    def ProcessFrame(self, lr):
        self.frame_device.upload(self.frame, self.stream)
        cv.cuda.resize(self.frame_device, (cols_big, rows_big), self.frame_device_big, stream=self.stream)
        self.bgmog2.apply(self.frame_device_big, lr, self.stream, self.fg_device_big)
        cv.cuda.resize(self.fg_device_big, self.fg_device.size(), self.fg_device, stream=self.stream)
        self.fg_device.download(self.stream, self.fg_host)
        self.stream.waitForCompletion()  # imidiate wait
        if (self.store_res):
            self.res.append(np.copy(self.fg_host))

    def Frame(self):
        return self.frame

proc_frame_cuda2 = ProcFrameCuda2(rows_small, cols_small, rows_big, cols_big, check_res)

#export
gpu_time_2, n_frames = ProcVid1(proc_frame_cuda2,lr)
print(f'GPU 2 (replacing the default stream): {n_frames} frames, {gpu_time_2:.2f} ms/frame')
print(f'Incremental speedup: {gpu_time_1/gpu_time_2:.2f}')
print(f'Speedup over GPU baseline: {gpu_time_0/gpu_time_2:.2f}')
print(f'Speedup over CPU: {cpu_time_1/gpu_time_2:.2f}')

if check_res: CheckFg(gpu_res,proc_frame_cuda2.res)

#export
# host mem not implemented, manually pin memory
class PinnedMem(object):
    def __init__(self, size, dtype=np.uint8):
        self.array = np.empty(size,dtype)
        cv.cuda.registerPageLocked(self.array)
        self.pinned = True
    def __del__(self):
        cv.cuda.unregisterPageLocked(self.array)
        self.pinned = False
    def __repr__(self):
        return f'pinned = {self.pinned}'

# export
class ProcFrameCuda3:
    def __init__(self, rows_small, cols_small, rows_big, cols_big, store_res=False):
        self.rows_small, self.cols_small, self.rows_big, self.cols_big = rows_small, cols_small, rows_big, cols_big
        self.store_res = store_res
        self.res = []
        self.bgmog2 = cv.cuda.createBackgroundSubtractorMOG2()
        self.stream = cv.cuda_Stream()
        self.frame = PinnedMem((rows_small, cols_small, 3))
        self.frame_device = cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC3)
        self.frame_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC3)
        self.fg_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC1)
        self.fg_device = cv.cuda_GpuMat(np.shape(frame)[0], np.shape(frame)[1], cv.CV_8UC1)
        self.fg_host = PinnedMem((rows_small, cols_small))

    def ProcessFrame(self, lr):
        self.frame_device.upload(self.frame.array, self.stream)
        cv.cuda.resize(self.frame_device, (cols_big, rows_big), self.frame_device_big, stream=self.stream)
        self.bgmog2.apply(self.frame_device_big, lr, self.stream, self.fg_device_big)
        cv.cuda.resize(self.fg_device_big, self.fg_device.size(), self.fg_device, stream=self.stream)
        self.fg_device.download(self.stream, self.fg_host.array)
        self.stream.waitForCompletion()  # imidiate wait
        if (self.store_res):
            self.res.append(np.copy(self.fg_host.array))

    def Frame(self):
        return self.frame.array

proc_frame_cuda3 = ProcFrameCuda3(rows_small, cols_small, rows_big, cols_big, check_res)

#export
gpu_time_3, n_frames = ProcVid1(proc_frame_cuda3,lr)
print(f'GPU 3 (overlap host and device - attempt 1): {n_frames} frames, {gpu_time_3:.2f} ms/frame')
print(f'Incremental speedup: {gpu_time_2/gpu_time_3:.2f}')
print(f'Speedup over GPU baseline: {gpu_time_0/gpu_time_3:.2f}')
print(f'Speedup over CPU: {cpu_time_1/gpu_time_3:.2f}')

if check_res: CheckFg(gpu_res,proc_frame_cuda3.res)

#export
def ProcVid2(proc_frame,lr,simulate=False):
    cap = cv.VideoCapture(vidPath)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        return
    n_frames = 0
    start = time.time()
    while(cap.isOpened()):
        ret,_ = cap.read(proc_frame.Frame())
        if ret == True:
            n_frames += 1
            if not simulate:
                proc_frame.ProcessFrame(lr)
        else:
            break
    proc_frame.Sync()
    end = time.time()
    cap.release()
    return (end - start)*1000/n_frames, n_frames;

# export
class ProcFrameCuda4:
    def __init__(self, rows_small, cols_small, rows_big, cols_big, store_res=False):
        self.rows_small, self.cols_small, self.rows_big, self.cols_big = rows_small, cols_small, rows_big, cols_big
        self.store_res = store_res
        self.res = []
        self.bgmog2 = cv.cuda.createBackgroundSubtractorMOG2()
        self.stream = cv.cuda_Stream()
        self.frame_num = 0
        self.i_writable_mem = 0
        self.frames_in = [PinnedMem((rows_small, cols_small, 3)), PinnedMem((rows_small, cols_small, 3))]
        self.frame_device = cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC3)
        self.frame_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC3)
        self.fg_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC1)
        self.fg_device = cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC1)
        self.fg_host = PinnedMem((rows_small, cols_small))

    def ProcessFrame(self, lr):
        self.frame_num += 1
        if (self.frame_num > 1):
            self.stream.waitForCompletion()  # wait after we have read the next frame
            if (self.store_res):
                self.res.append(np.copy(self.fg_host.array))
        self.frame_device.upload(self.frames_in[self.i_writable_mem].array, self.stream)
        cv.cuda.resize(self.frame_device, (cols_big, rows_big), self.frame_device_big, stream=self.stream)
        self.bgmog2.apply(self.frame_device_big, lr, self.stream, self.fg_device_big)
        cv.cuda.resize(self.fg_device_big, self.fg_device.size(), self.fg_device, stream=self.stream)
        self.fg_device.download(self.stream, self.fg_host.array)

    def Frame(self):
        self.i_writable_mem = (self.i_writable_mem + 1) % len(self.frames_in)
        return self.frames_in[self.i_writable_mem].array

    def Sync(self):
        self.stream.waitForCompletion()
        if (self.store_res):
            self.res.append(np.copy(self.fg_host.array))

proc_frame_cuda4 = ProcFrameCuda4(rows_small, cols_small, rows_big, cols_big, check_res)

#export
gpu_time_4, n_frames = ProcVid2(proc_frame_cuda4,lr)
print(f'GPU 4 (overlap host and device - attempt 2): {n_frames} frames, {gpu_time_4:.2f} ms/frame')
print(f'Incremental speedup: {gpu_time_3/gpu_time_4:.2f}')
print(f'Speedup over GPU baseline: {gpu_time_0/gpu_time_4:.2f}')
print(f'Speedup over CPU: {cpu_time_1/gpu_time_4:.2f}')

if check_res: CheckFg(gpu_res,proc_frame_cuda4.res)

# export
class ProcFrameCuda5:
    def __init__(self, rows_small, cols_small, rows_big, cols_big, store_res=False):
        self.rows_small, self.cols_small, self.rows_big, self.cols_big = rows_small, cols_small, rows_big, cols_big
        self.store_res = store_res
        self.res = []
        self.bgmog2 = cv.cuda.createBackgroundSubtractorMOG2()
        self.stream = cv.cuda_Stream()
        self.frame_num = 0
        self.i_writable_mem = 0
        self.frames_in = [PinnedMem((rows_small, cols_small, 3)), PinnedMem((rows_small, cols_small, 3))]
        self.frame_device = cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC3)
        self.frame_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC3)
        self.fg_device_big = cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC1)
        self.fg_device = cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC1)
        self.fg_host = PinnedMem((rows_small, cols_small))

    def ProcessFrame(self, lr):
        self.frame_num += 1
        if (self.frame_num > 1):
            self.stream.waitForCompletion()  # wait after we have read the next frame
            if (self.store_res):
                self.res.append(np.copy(self.fg_host.array))
        self.frame_device.upload(self.frames_in[self.i_writable_mem].array, self.stream)
        cv.cuda.resize(self.frame_device, (cols_big, rows_big), self.frame_device_big, stream=self.stream)
        self.bgmog2.apply(self.frame_device_big, lr, self.stream, self.fg_device_big)
        cv.cuda.resize(self.fg_device_big, self.fg_device.size(), self.fg_device, stream=self.stream)
        self.fg_device.download(self.stream, self.fg_host.array)
        self.stream.queryIfComplete()  # kick WDDM

    def Frame(self):
        self.i_writable_mem = (self.i_writable_mem + 1) % len(self.frames_in)
        return self.frames_in[self.i_writable_mem].array

    def Sync(self):
        self.stream.waitForCompletion()
        if (self.store_res):
            self.res.append(np.copy(self.fg_host.array))

proc_frame_cuda5 = ProcFrameCuda5(rows_small, cols_small, rows_big, cols_big, check_res)

#export
gpu_time_5, n_frames = ProcVid2(proc_frame_cuda5,lr)
print(f'GPU 5 (overlap host and device - attempt 3): {n_frames} frames, {gpu_time_5:.2f} ms/frame')
print(f'Incremental speedup: {gpu_time_4/gpu_time_5:.2f}')
print(f'Speedup over GPU baseline: {gpu_time_0/gpu_time_5:.2f}')
print(f'Speedup over CPU: {cpu_time_1/gpu_time_5:.2f}')

if check_res:  CheckFg(gpu_res,proc_frame_cuda5.res)

# export
class SyncType():
    none = 1
    soft = 2
    hard = 3

class ProcFrameCuda6:
    def __init__(self, rows_small, cols_small, rows_big, cols_big, n_streams, store_res=False, sync=SyncType.soft,
                 device_timer=False):
        self.rows_small, self.cols_small, self.rows_big, self.cols_big = rows_small, cols_small, rows_big, cols_big
        self.n_streams = n_streams
        self.store_res = store_res
        self.sync = sync
        self.bgmog2 = cv.cuda.createBackgroundSubtractorMOG2()
        self.frames_device = []
        self.frames_device_big = []
        self.fgs_device_big = []
        self.fgs_device = []
        self.fgs_small = []
        self.streams = []
        self.frames = []
        self.InitMem()
        self.InitStreams()
        self.res = []
        self.i_stream = 0
        self.n_frames = 0
        self.i_writable_mem = 0
        self.device_timer = device_timer
        if self.device_timer:
            self.events_start = []
            self.events_stop = []
            self.InitEvents()
            self.device_time = 0

    def InitMem(self):
        for i in range(0, self.n_streams + 1):
            self.frames.append(PinnedMem((rows_small, cols_small, 3)))

        for i in range(0, self.n_streams):
            self.frames_device.append(cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC3))
            self.frames_device_big.append(cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC3))
            self.fgs_device_big.append(cv.cuda_GpuMat(rows_big, cols_big, cv.CV_8UC1))
            self.fgs_device.append(cv.cuda_GpuMat(rows_small, cols_small, cv.CV_8UC1))
            self.fgs_small.append(PinnedMem((rows_small, cols_small)))

    def InitStreams(self):
        for i in range(0, self.n_streams):
            if self.sync == SyncType.hard:
                self.streams.append(cv.cuda.Stream_Null())
            elif self.sync == SyncType.soft:
                self.streams.append(cv.cuda_Stream())

    def InitEvents(self):
        for i in range(0, self.n_streams):
            self.events_start.append(cv.cuda_Event())
            self.events_stop.append(cv.cuda_Event())

    def IncStream(self):
        self.i_stream = (self.i_stream + 1) % self.n_streams

    def ProcessFrame(self, lr):
        self.n_frames += 1
        i = self.i_stream
        self.IncStream()
        stream = self.streams[i]
        if (self.n_frames > self.n_streams and self.sync != SyncType.none):
            stream.waitForCompletion()  # wait once both streams are used
            if self.device_timer:  self.device_time += cv.cuda_Event.elapsedTime(self.events_start[i],
                                                                                 self.events_stop[i])
            if (self.store_res):
                self.res.append(np.copy(self.fgs_small[i].array))
        if self.device_timer: self.events_start[i].record(stream)
        self.frames_device[i].upload(self.frames[self.i_writable_mem].array, stream)
        cv.cuda.resize(self.frames_device[i], (cols_big, rows_big), self.frames_device_big[i], stream=stream)
        self.bgmog2.apply(self.frames_device_big[i], lr, stream, self.fgs_device_big[i])
        cv.cuda.resize(self.fgs_device_big[i], self.fgs_device[i].size(), self.fgs_device[i], stream=stream)
        self.fgs_device[i].download(stream, self.fgs_small[i].array)
        if self.device_timer: self.events_stop[i].record(stream)
        stream.queryIfComplete()  # kick WDDM

    def Frame(self):
        self.i_writable_mem = (self.i_writable_mem + 1) % len(self.frames)
        return self.frames[self.i_writable_mem].array

    def Sync(self):
        # sync on last frames
        if (self.sync == SyncType.none):
            return

        for i in range(0, self.n_streams):
            if (not self.streams[self.i_stream].queryIfComplete()):
                self.streams[self.i_stream].waitForCompletion()
            if (self.store_res):
                self.res.append(np.copy(self.fgs_small[self.i_stream].array))
            self.IncStream()

    def FrameTimeMs(self):
        if self.device_timer:
            return self.device_time / self.n_frames
        else:
            return 0

proc_frame_cuda6 = ProcFrameCuda6(rows_small, cols_small, rows_big, cols_big, 2, check_res, SyncType.soft)

#export
gpu_time_6, n_frames = ProcVid2(proc_frame_cuda6,lr)
print(f'GPU 6 (multiple streams): {n_frames} frames, {gpu_time_6:.2f} ms/frame')
print(f'Incremental speedup: {gpu_time_5/gpu_time_6:.2f}')
print(f'Speedup over GPU baseline: {gpu_time_0/gpu_time_6:.2f}')
print(f'Speedup over CPU: {cpu_time_1/gpu_time_6:.2f}')

if check_res: CheckFg(gpu_res,proc_frame_cuda6.res)

#export
proc_frame_cuda7 = ProcFrameCuda6(rows_small,cols_small,rows_big,cols_big,2,check_res,SyncType.soft,True)
ProcVid2(proc_frame_cuda7,lr)
print(f'Mean times calculated over {n_frames} frames:')
print(f'Time to process each frame on the device: {proc_frame_cuda7.FrameTimeMs():.2f} ms/frame')
print(f'Time to process each frame (host/device): {gpu_time_6:.2f} ms/frame')
print(f'-> Gain from memcpy/kernel overlap if device is saturated: {proc_frame_cuda7.FrameTimeMs()-gpu_time_6:.2f} ms/frame')
hostTime, n_frames = ProcVid2(proc_frame_cuda6, lr, True)
print(f'Time to read and decode each frame on the host: {hostTime:.2f} ms/frame')
print(f'-> Total processing time host + device: {proc_frame_cuda7.FrameTimeMs()+hostTime:.2f} ms/frame')
print(f'-> Gain from host/device overlap: {proc_frame_cuda7.FrameTimeMs()+hostTime - gpu_time_6:.2f} ms/frame')
print(f'-> Currently waisted time on host: {gpu_time_6-hostTime:.2f} ms/frame')
