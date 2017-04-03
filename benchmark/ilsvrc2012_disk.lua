--[[
    Benchmarking script to analyse the loading speed by loading metadata from disk.

    The intent for this code is to show that loading metadata from disk has an insignificant
    overhead when retrieving metadata stored in RAM.
]]


require 'paths'
require 'torch'
require 'string'
require 'xlua'
local tnt = require 'torchnet'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)

--------------------------------------------------------------------------------
-- Load configurations (data, model, criterion, optimState, logger)
--------------------------------------------------------------------------------

local opt = {
    dataset = 'ilsvrc2012',
    expDir = './cache',
    manualSeed = 2,
    nThreads = 4,
    imageSize = 256,
    cropSize = 224,
    batchSize = 128,
    nsamples = 1000
}

local stats = torch.load(paths.concat(opt.expDir, opt.dataset, 'cache.t7'))
opt.meanstd = {mean = stats.meanstd.mean, std = stats.meanstd.std}

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end

-- number of samples to fetch
local n_samples = opt.nsamples


--------------------------------------------------------------------------------
-- Setup data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nThreads,
      ordered=true,
      init    = function(threadid)
                  require 'torch'
                  require 'torchnet'
                  require 'image'
                  t = require 'transforms'
                  torch.manualSeed(threadid+opt.manualSeed)
                end,
      closure = function()

         -- get data loader function
         local data_loader = require 'data'
         local GetDataFn, data_size = data_loader(opt.dataset, mode)

         -- setup dataset iterator
         local list_dataset = tnt.ListDataset{  -- replace this by your own dataset
            list = torch.range(1, n_samples):long(),
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
                  or mode == 'test' and
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
-- Setup timer
--------------------------------------------------------------------------------

local dataTimer = torch.Timer()


-----------------------------------------------------------
-- Iterate over n samples (train + test)
-----------------------------------------------------------

for _, mode in pairs({'train', 'test'}) do
    print('\n************************************************')
    print('Benchmarking loading metadata from disk: ' .. mode)
    print('************************************************\n')

    -- get data iterator
    local iterator = getIterator(mode)

    local sample_time = {}
    for i=1, n_samples do
        xlua.progress(i, n_samples)
        dataTimer:reset()
        local sample = iterator()
        table.insert(sample_time, dataTimer:time().real)
        collectgarbage()
    end

    -- display timings info for the set
    print(('> Average data loading time (batchsize=%d): %0.5f'):format(opt.batchSize, torch.DoubleTensor(sample_time):mean()))
    print(('> Min data loading time (batchsize=%d): %0.5f'):format(opt.batchSize, torch.DoubleTensor(sample_time):min()))
    print(('> Max data loading time (batchsize=%d): %0.5f'):format(opt.batchSize, torch.DoubleTensor(sample_time):max()))
    print(('> Total data loading time (batchsize=%d): %0.5f'):format(opt.batchSize, torch.DoubleTensor(sample_time):sum()))
end

print('\n==> Script complete.')
