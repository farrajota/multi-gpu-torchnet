local function createModel(nClasses)
   require 'cudnn'
   
   local nClasses = nClasses or 1000
   
   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            features:add(conv3)
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end
   
   local classifier = nn.Sequential()
   classifier:add(nn.SpatialAveragePooling(7, 7, 1, 1))
   classifier:add(nn.SpatialBatchNormalization(512, 1e-3))
   classifier:add(nn.View(512):setNumInputDims(3))
   classifier:add(nn.Linear(512, nClasses))
   --classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)
   model.imageSize = 256
   model.imageCrop = 224

   return model
end

----------------------------

return createModel