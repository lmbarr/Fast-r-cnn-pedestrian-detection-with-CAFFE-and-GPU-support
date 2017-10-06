################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Candidatos.cpp \
../Clasificador.cpp \
../Detector.cpp \
../HeadDetector.cpp \
../NonMaximaSupression.cpp \
../RegionGrowing.cpp 

OBJS += \
./Candidatos.o \
./Clasificador.o \
./Detector.o \
./HeadDetector.o \
./NonMaximaSupression.o \
./RegionGrowing.o 

CPP_DEPS += \
./Candidatos.d \
./Clasificador.d \
./Detector.d \
./HeadDetector.d \
./NonMaximaSupression.d \
./RegionGrowing.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/cuda/include -I/home/ubuntu/fast-rcnn/caffe-fast-rcnn/include -I/media/ubuntu/Tesis/opencv/opencv-3.1.0/archivos/include/opencv -I/home/ubuntu/fast-rcnn/caffe-fast-rcnn/distribute/include  -I/media/ubuntu/Tesis/opencv/opencv-3.1.0/archivos/include -O0 -g3 -Wall -c -fmessage-length=0  -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


