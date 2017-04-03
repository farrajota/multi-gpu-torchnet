--[[
    List of available models for imagenet.
]]

return {
   -- AlexNet
   alexnet = paths.dofile('alexnet.lua'),
   alexnetowt = paths.dofile('alexnetowt.lua'),
   alexnetowtbn = paths.dofile('alexnetowtbn.lua'),
   alexnetbn = paths.dofile('alexnetbn.lua'),
   
   -- GoogleNet (v1)
   googlenet = paths.dofile('googlenet.lua'),
   
   -- Network-in-network
   ninbn = paths.dofile('ninbn.lua'),
   
   -- Overfeat
   overfeat = paths.dofile('overfeat.lua'),
   
   -- SqueezeNet
   squeezenet = paths.dofile('squeezenet.lua'),
   
   -- Oxford VGG
   vgg = paths.dofile('vgg.lua'),
   vggbn = paths.dofile('vggbn.lua'),
   vggbnv2 = paths.dofile('vggbnv2.lua')
}