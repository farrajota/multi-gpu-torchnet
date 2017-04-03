local function createModel(nClasses)
   local nClasses = nClasses or 1000
   
   local features = nn.Sequential()

   features:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(nn.SpatialConvolution(96, 256, 5, 5, 1, 1))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
   features:add(nn.ReLU(true))

   features:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
   features:add(nn.ReLU(true))

   features:add(nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(1024*5*5))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(1024*5*5, 3072))
   classifier:add(nn.Threshold(0, 1e-6))

   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(3072, 4096))
   classifier:add(nn.Threshold(0, 1e-6))

   classifier:add(nn.Linear(4096, nClasses))
   --classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.2 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 256
   model.imageCrop = 224
   return model
end

----------------------------

return createModel