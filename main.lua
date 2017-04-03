--[[
    *** Main script ***   Train a model on the ilsvrc2012/cifar10/cifar100dataset.
--]]

require 'paths'
require 'torch'
require 'string'
local tnt = require 'torchnet'


--------------------------------------------------------------------------------
-- Load configurations (data, model, criterion, optimState, logger)
--------------------------------------------------------------------------------

require 'configs'
local lopt = opt

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end


--------------------------------------------------------------------------------
-- Setup data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nThreads,
      init    = function(threadid)
                  require 'torch'
                  require 'torchnet'
                  require 'image'
                  t = require 'transforms'
                  opt = lopt
                  torch.manualSeed(threadid+opt.manualSeed)
                end,
      closure = function()

         -- get data loader function
         local data_loader = require 'data'
         local GetDataFn, data_size = data_loader(opt.dataset, mode)

         -- setup dataset iterator
         local list_dataset = tnt.ListDataset{  -- replace this by your own dataset
            list = torch.range(1, data_size):long(),
            load = function(idx)
               local data = GetDataFn()
               return {
                   input  = data[1],
                   target = data[2],
               }
            end
          }

         return list_dataset
            :shuffle()
            :transform{
              input = mode == 'train' and
                     tnt.transform.compose{
                        t.Fix(),
                        t.Scale(opt.imageSize),
                        t.RandomCrop(opt.cropSize),
                        t.ColorNormalize({mean = opt.meanstd.mean, std = opt.meanstd.std}),
                        t.HorizontalFlip(0.5),
                        t.ColorJitter({
                           brightness = 0.4,
                           contrast = 0.4,
                           saturation = 0.4,
                        })
                     }
                  or (mode == 'test' or mode == 'val') and
                     tnt.transform.compose{
                        t.Fix(),
                        t.Scale(opt.imageSize),
                        t.CenterCrop(opt.cropSize),
                        t.ColorNormalize({mean = opt.meanstd.mean, std = opt.meanstd.std})
                     }
            }
            :batch(opt.batchSize, 'include-last')
      end,
   }
end

--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

local timers = {
   batchTimer = torch.Timer(),
   dataTimer = torch.Timer(),
   epochTimer = torch.Timer(),
}

local meters = {
   conf = tnt.ConfusionMeter{k = opt.nClasses},
   val = tnt.AverageValueMeter(),
   train = tnt.AverageValueMeter(),
   train_clerr = tnt.ClassErrorMeter{topk = {1,5},accuracy=true},
   clerr = tnt.ClassErrorMeter{topk = {1,5},accuracy=true},
   ap = tnt.APMeter(),
}

function meters:reset()
   self.conf:reset()
   self.val:reset()
   self.train:reset()
   self.train_clerr:reset()
   self.clerr:reset()
   self.ap:reset()
end

local loggers = {
   test = optim.Logger(paths.concat(opt.save,'test.log')),
   train = optim.Logger(paths.concat(opt.save,'train.log')),
   full_train = optim.Logger(paths.concat(opt.save,'full_train.log')),
}

loggers.test:setNames{'Test Loss', 'Test acc.', 'Test mAP'}
loggers.train:setNames{'Train Loss', 'Train acc.'}
loggers.full_train:setNames{'Train Loss'}

loggers.test.showPlot = false
loggers.train.showPlot = false
loggers.full_train.showPlot = false


-- set up training engine:
local engine = tnt.OptimEngine()

engine.hooks.onStart = function(state)
   if state.training then
      if opt.iniEpoch>1 then
         state.epoch = math.max(opt.iniEpoch, state.epoch)
      end
   end
end

engine.hooks.onStartEpoch = function(state)
   print('\n***********************')
   print('Start Train epoch=' .. state.epoch+1)
   print('***********************')
   timers.epochTimer:reset()
   state.config = optimStateFn(state.epoch+1)
end

engine.hooks.onForwardCriterion = function(state)
   if state.training then
      meters.train:add(state.criterion.output)
      meters.train_clerr:add(state.network.output,state.sample.target)
      if opt.verbose then
         print(string.format('epoch[%d/%d][%d/%d][batch=%d] - loss: %2.4f; top-1 err: ' ..
                             '%2.2f; top-5 err: %2.2f; lr = %2.2e;  DataLoadingTime: %0.5f; ' ..
                             'forward-backward time: %0.5f', state.epoch+1, state.maxepoch,
                             state.t+1, nBatchesTrain, opt.batchSize, meters.train:value(),
                             100-meters.train_clerr:value{k = 1}, 100-meters.train_clerr:value{k = 5},
                             state.config.learningRate, timers.dataTimer:time().real,
                             timers.batchTimer:time().real))
     else
        xlua.progress(state.t+1, nBatchesTrain)
     end

      loggers.full_train:add{state.criterion.output}
   else
      meters.conf:add(state.network.output,state.sample.target)
      meters.clerr:add(state.network.output,state.sample.target)
      meters.val:add(state.criterion.output)
      local tar = torch.ByteTensor(#state.network.output):fill(0)
      for k=1,state.sample.target:size(1) do
         local id = state.sample.target[k]:squeeze()
         tar[k][id]=1
      end
      meters.ap:add(state.network.output,tar)
   end
end

-- copy sample to GPU buffer:
local inputs = cast(torch.Tensor())
local targets = cast(torch.Tensor())
engine.hooks.onSample = function(state)
   inputs:resize(state.sample.input:size() ):copy(state.sample.input)
   targets:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input  = inputs
   state.sample.target = targets
   timers.dataTimer:stop()
   timers.batchTimer:reset()
end

engine.hooks.onForward = function(state)
   if not state.training then
      xlua.progress(state.t, nBatchesTest)
   end
end

engine.hooks.onUpdate = function(state)
    timers.dataTimer:reset()
    timers.dataTimer:resume()
end

engine.hooks.onEndEpoch = function(state)
   print("Epoch Train Loss:" ,meters.train:value(),"Total Epoch time: ",timers.epochTimer:time().real)
   -- measure test loss and error:
   loggers.train:add{meters.train:value(),meters.train_clerr:value()[1]}
   meters:reset()
   state.t = 0
   print('\n***********************')
   print('Test network (epoch=' .. state.epoch .. ')')
   print('***********************')
   engine:test{
      network   = model,
      iterator  = getIterator('test'),
      criterion = criterion,
   }

   loggers.test:add{meters.val:value(),meters.clerr:value()[1],meters.ap:value():mean()}
   print("Validation Loss" , meters.val:value())
   print("Accuracy: Top 1%", meters.clerr:value{k = 1})
   print("Accuracy: Top 5%", meters.clerr:value{k = 5})
   print("mean AP:",meters.ap:value():mean())
   log(state.network, state.config, meters, loggers, state.epoch)
   print("Testing Finished")
   timers.epochTimer:reset()
   state.t = 0
end


--------------------------------------------------------------------------------
-- Train the model
--------------------------------------------------------------------------------

print('==> Train network model')
engine:train{
   network   = model,
   iterator  = getIterator('train'),
   criterion = criterion,
   optimMethod = optim.sgd,
   config = {
     learningRate = opt.LR,
     momentum = opt.momentum,
     weightDecay = opt.weightDecay,
   },
   maxepoch = opt.nEpochs
}

print('==> Script complete.')