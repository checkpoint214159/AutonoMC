FROM minedojo/minedojo

RUN sudo mkdir /AutonoMC

COPY troublesome_build /AutonoMC/troublesome_build

RUN sudo cp /AutonoMC/troublesome_build/build.gradle /opt/conda/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/

RUN python /AutonoMC/troublesome_build/test.py