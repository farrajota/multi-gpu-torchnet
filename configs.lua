--[[
    Setup necessary dependencies/packages and data.
]]


require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'image'


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

local opts = require 'opts'
opt = opts.parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')

print('Saving everything to: ' .. opt.save)
paths.mkdir(opt.save)
torch.save(paths.concat(opt.save, 'options.t7'), opt)



-- number of total batches the train and test sets
do
  local dbc = require 'dbcollection.manager'
  local dbdataset = dbc.load{name = opt.dataset, data_dir=opt.data_dir}

  nBatchesTrain = math.floor(dbdataset:size('train')[1] / opt.batchSize)
  if opt.dataset == 'ilsvrc2012' then
     nBatchesTest = math.floor(dbdataset:size('val')[1] / opt.batchSize)
  else
     nBatchesTest = math.floor(dbdataset:size('test')[1] / opt.batchSize)
  end
end


--------------------------------------------------------------------------------
-- Check if cache is available
--------------------------------------------------------------------------------

if not paths.filep(paths.concat(opt.expDir,'cache.t7')) then
   paths.dofile('statistics.lua')() -- process dataset's mean/std
end

local stats = torch.load(paths.concat(opt.expDir, 'cache.t7'))
opt.meanstd = {mean = stats.meanstd.mean, std = stats.meanstd.std}
opt.nClasses = stats.nClasses


--------------------------------------------------------------------------------
-- Load network + criterion
--------------------------------------------------------------------------------

local net = require 'model'
criterion = nn.CrossEntropyCriterion()


--------------------------------------------------------------------------------
-- Use GPU or CPU
--------------------------------------------------------------------------------

if opt.GPU >= 1 then
   -- Use GPU
   print(('Running on GPU: (number of GPUs used: [%d])'):format(opt.nGPU))
   require 'cutorch'
   require 'cunn'

   cutorch.setDevice(opt.GPU) -- by default, use GPU 1

   -- set to cuda
   net:cuda()
   criterion:cuda()

   -- require cudnn if available
   if opt.backend == 'cudnn' and pcall(require, 'cudnn') then
      cudnn.convert(net, cudnn)
      cudnn.benchmark = true
      print('Network has', #net:findModules'cudnn.SpatialConvolution', 'cudnn convolutions')
   end
   opt.dataType = 'torch.CudaTensor'
else
   -- Use CPU
   print(('Running on CPU'):format())
   -- set to float
   net:float()
   criterion:float()
   opt.dataType = 'torch.FloatTensor'
end
print(net)


--------------------------------------------------------------------------------
-- Optimize networks memory usage
--------------------------------------------------------------------------------

if opt.optimize then
   -- for memory optimizations and graph generation
   local optnet = require 'optnet'

   local sample_input = torch.randn(math.max(1,math.floor(opt.batchSize/4)), 3, opt.cropSize, opt.cropSize):float()
   if opt.GPU>=1 then sample_input=sample_input:cuda() end
   optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
end

-- Use multiple gpus
model = nn.Sequential()
if opt.GPU >= 1 and opt.nGPU > 1 then
  local utils = require 'utils'
  model:add(utils.makeDataParallelTable(net, opt.nGPU))
else
  model:add(net)
end


--------------------------------------------------------------------------------
-- Config optim states into a function
--------------------------------------------------------------------------------

function optimStateFn(epoch)
  local regimes = {
      -- start, end,    LR,   WD,
      {  1,     18,   1e-2,   5e-4, },
      { 19,     29,   5e-3,   5e-4  },
      { 30,     43,   1e-3,   0 },
      { 44,     52,   5e-4,   0 },
      { 53,    1e8,   1e-4,   0 },
  }

  for i=1, #regimes do
    if epoch >= regimes[i][1] and epoch <= regimes[i][2] then
      return {
          learningRate = regimes[i][3],
          learningRateDecay = 0.0,
          dampening = 0.0,
          momentum = opt.momentum,
          weightDecay = regimes[i][4],
      }
    end
  end

  return {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      dampening = 0.0,
      momentum = opt.momentum,
      weightDecay = opt.weightDecay,
  }
end


--------------------------------------------------------------------------------
-- Load logger function
--------------------------------------------------------------------------------

log = require 'logger'