--[[
    Benchmarking script to analyse the loading speed by loading metadata directly from RAM memory.

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
    dataset = 'cifar10',
    manualSeed = 2,
    nThreads = 4,
    imageSize = 48,
    cropSize = 32,
    batchSize = 128,
    nsamples = 1000,
    meanstd = {
        mean = {0.25, 0.25, 0.25},
        std = {0.01, 0.01, 0.01}
    }
}

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end

-- number of samples to fetch
local n_samples = opt.nsamples


-----------------------------------------------------------
-- Move metadata from disk to memory
-----------------------------------------------------------

print('Loading data from file to ram ...')
local dbc = require 'dbcollection.manager'
local dbloader = dbc.load(opt.dataset)
local num_classes = dbloader:size('train', 'classes')[1]
local string_ascii = require 'dbcollection.utils.string_ascii'
local ascii2str = string_ascii.convert_ascii_to_str


-- train data
local train_size = dbloader:size('train')[1]
local train_images = dbloader:get('train', 'images'):transpose(3,4):transpose(2,3):float():div(255)
local train_class_ids = dbloader:object('train'):select(2,2):long():add(1)

-- val data
local test_size = dbloader:size('test')[1]
local test_images = dbloader:get('test', 'images'):transpose(3,4):transpose(2,3):float():div(255)
local test_class_ids = dbloader:object('train'):select(2,2):long():add(1)

local metadata = {
    train = {
        images = train_images,
        class_id = train_class_ids,
        size = train_size
    },
    test = {
        images = test_images,
        class_id = test_class_ids,
        size = test_size
    },
}


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

         local data = metadata[mode]

         local function fetch_data(idx)
            -- get random image from the list
            return {data.images[idx], data.class_id[idx]}
         end

         -- setup dataset iterator
         local list_dataset = tnt.ListDataset{
            list = torch.range(1, data.size):long(),
            load = function(idx)
               local data = fetch_data(idx)
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
-- Iterate over n samples (train + val)
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
