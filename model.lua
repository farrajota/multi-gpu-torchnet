--[[
    Load network into memory.
]]

require 'nn'
require 'nngraph'

local model
if opt.continue and paths.filep(paths.concat(opt.save, 'last_snapshot.t7')) then
   local prevModelFilename
   local epoch = torch.load(paths.concat(opt.save, 'last_snapshot.t7'))
   if paths.filep(opt.save, 'last_snapshot.t7') then
      prevModelFilename = opt.save .. '/model.t7'
   else
      prevModelFilename = opt.save .. '/model_' .. epoch .. '.t7'
   end

   print('Continue training at epoch ' .. epoch)
   print('==> Loading model from file: ' .. prevModelFilename)
   model = torch.load(prevModelFilename)
   opt.iniEpoch = epoch

else
   -- return model instead of function call (see models/alexnetowtbn)
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   local model_list = require 'models'
   model = model_list[opt.netType](opt.nClasses)
   opt.iniEpoch=1

   if opt.backend ~= 'cudnn' and opt.backend ~= 'nn' then
      error('Unsupported backend: ' .. opt.backend)
   end

end

return model
