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


-----------------------------------------------------------
-- Move metadata from disk to memory
-----------------------------------------------------------

print('Loading data from file to ram ...')
local dbc = require 'dbcollection.manager'
local dbloader = dbc.load(opt.dataset)
local num_classes = dbloader:size('train', 'classes')[1]
local string_ascii = require 'dbcollection.utils.string_ascii'
local ascii2str = string_ascii.convert_ascii_to_str

local function add_path(fname)
    local out = {}
    for i=1, #fname do
        table.insert(out, paths.concat(dbloader.data_dir, fname[i]))
    end
    return out
end

-- train data
local train_objs_ids = dbloader:object('train')
local train_filenames = add_path(ascii2str(dbloader:get('train', 'image_filenames')))
local train_class_ids = train_objs_ids:select(2,2):long()

-- val data
local val_objs_ids = dbloader:object('val')
local val_filenames = add_path(ascii2str(dbloader:get('val', 'image_filenames')))
local val_class_ids = val_objs_ids:select(2,2):long()

local metadata = {
    train = {
        filename = train_filenames,
        class_id = train_class_ids,
        size = train_objs_ids:size(1)
    },
    val = {
        filename = val_filenames,
        class_id = val_class_ids,
        size = val_objs_ids:size(1)
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
            local idx = torch.random(1, data.size)
            return {image.load(data.filename[idx], 3, 'float'), data.class_id[idx]}
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
                  or mode == 'val' and
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

for _, mode in pairs({'train', 'val'}) do
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
