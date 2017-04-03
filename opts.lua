--[[
    Input options selection/parser.
]]


local function parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 (Torchnet) Imagenet Training script')
   cmd:text()
   cmd:text('Options:')
   ------------ General options --------------------
   cmd:option('-dataset', 'ilsvrc2012', 'Dataset choices: ilsvrc2012 | cifar10 | cifar 100 | mnist')
   cmd:option('-expID',    'alexnet55', 'Experiment ID')
   cmd:option('-expDir',     './cache', 'subdirectory in which to save/log experiments')
   cmd:option('-data_dir',          '', 'Directory to load/download the dataset\'s files to/from disk.')
   cmd:option('-manualSeed',         2, 'Manually set RNG seed')
   cmd:option('-GPU',                1, 'Default preferred GPU (-1 sets to CPU mode)')
   cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
   cmd:option('-backend',      'cudnn', 'Options: cudnn | nn')
   ------------- Data options ------------------------
   cmd:option('-nThreads',           4, 'number of data loading threads')
   cmd:option('-imageSize',        256, 'Smallest side of the resized image')
   cmd:option('-cropSize',         224, 'Height and Width of image crop to be used as input layer')
   ------------- Training options --------------------
   cmd:option('-nEpochs',           55, 'Number of total epochs to run')
   cmd:option('-batchSize',        128, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-verbose',       'true', 'true- detailed information | false - progress bar')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              1e-2, 'learning rate; if set, overrides default LR/WD recipe')
   cmd:option('-momentum',         0.9, 'momentum')
   cmd:option('-weightDecay',     5e-4, 'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-snapshot',           5, 'How often to take a snapshot of the model (>0 => saves into a file at a certain epoch; =0 => never (saves the model only in the last epoch); <0 => save into a single file at a certain epoch)')
   cmd:option('-optimize',      'true', 'Optimize the networks memory usage.')
   cmd:option('-netType',  'alexnetbn', 'Options: check the models folder for all available models.')
   cmd:option('-retrain',       'none', 'Provide path to model to retrain with')
   cmd:option('-continue',     'false', 'Provide path to an optimState to reload from')
   cmd:text()

   ------------------------------------------------------------
   local function Str2BoolFn(input) -- converts string to boolean
      assert(input)
      if type(input) == 'boolean' then
         return input
      end

      if string.lower(input) == 'true' then
         return true
      else
         return false
      end
   end
   ------------------------------------------------------------

  -- parse options
   local opt = cmd:parse(arg or {})
   opt.expDir = paths.concat(opt.expDir, opt.dataset)
   opt.save = paths.concat(opt.expDir, opt.expID)

   opt.optimize = Str2BoolFn(opt.optimize)
   opt.verbose = Str2BoolFn(opt.verbose)
   opt.continue = Str2BoolFn(opt.continue)

   return opt
end

return {
   parse = parse
}
