--[[
    Logger functions for easier logging data management.
]]


local utils = paths.dofile('utils.lua')

------------------------------------

local function storeModel(model, optimState, epoch, flag)
   if flag then
      print('Saving model snapshot to: ' .. paths.concat(opt.save,'model_' .. epoch ..'.t7'))
      utils.saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:clearState())

      torch.save(paths.concat(opt.save,'last_epoch.t7'), epoch)
      torch.save(paths.concat(opt.save,'optim_' .. epoch ..'.t7'), optimstate)
      torch.save(paths.concat(opt.save,'meters_' .. epoch ..'.t7'), meters)

   else
      print('Saving model snapshot to: ' .. paths.concat(opt.save,'model.t7'))
      utils.saveDataParallel(paths.concat(opt.save, 'model.t7'), model:clearState())

      torch.save(paths.concat(opt.save,'last_epoch.t7'), epoch)
      torch.save(paths.concat(opt.save,'optim.t7'), optimstate)
      torch.save(paths.concat(opt.save,'meters.t7'),meters)

   end
end

------------------------------------

local function logging(model, optimState, meters, loggers, epoch)
   require 'image'
   meters.conf.normalized = true
   image.save(paths.concat(opt.save,'confusion_' .. epoch ..'.jpg'),
               image.scale(meters.conf:value():float(),1000,1000,'simple'))
   meters.conf.normalized = false

   loggers.test:style{'+-','+-','+-'}
   loggers.test:plot()

   loggers.train:style{'+-','+-'}
   loggers.train:plot()

   loggers.full_train:style{'+-'}
   loggers.full_train:plot()

   -- store model snapshot
   if opt.snapshot > 0 then
     if epoch%opt.snapshot == 0 then
        storeModel(model.modules[1], optimState, epoch, true)
     end

   elseif opt.snapshot < 0 then
     if epoch%math.abs(opt.snapshot) == 0 then
        storeModel(model.modules[1], optimState, epoch, false)
     end
   else
      -- save only at the last epoch
      if epoch == opt.nEpochs then
        storeModel(model.modules[1], optimState, epoch, false)
      end
   end

end

------------------------------------

return logging
