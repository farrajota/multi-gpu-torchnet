--[[
    Functions to load data from various datasets (cifar/mnist/ilsvrc2012).
]]


require 'image'
local string_ascii = require 'dbcollection.utils.string_ascii'
local ascii2str = string_ascii.convert_ascii_to_str


local function loader_mnist(dbloader, num_classes, mode)
    local function fetch_data()
        -- get random class index
        local classID = torch.random(1, num_classes)
        local imgs_class = dbloader:get(mode, 'list_images_per_class', classID)

        -- remove negative indexes
        imgs_class = imgs_class[imgs_class:ge(0)]

        -- get random image index
        local random_idx = torch.random(1, imgs_class:size(1))
        local img_idx = imgs_class[random_idx] + 1 -- lua is 1-indexed

        -- fetch image filename
        local img = dbloader:get(mode, 'images', img_idx):float():div(255):repeatTensor(3,1,1)

        -- output data
        return {img, torch.LongTensor{classID}}
    end

    return fetch_data, dbloader:size(mode)[1]
end


local function loader_cifar(dbloader, num_classes, mode)
    local function fetch_data()
        -- get random class index
        local classID = torch.random(1, num_classes)
        local imgs_class = dbloader:get(mode, 'list_images_per_class', classID)

        -- remove negative indexes
        imgs_class = imgs_class[imgs_class:ge(0)]

        -- get random image index
        local random_idx = torch.random(1, imgs_class:size(1))
        local img_idx = imgs_class[random_idx] + 1 -- lua is 1-indexed

        -- fetch image filename
        local img = dbloader:get(mode, 'images', img_idx):transpose(3,4):transpose(2,3):float():div(255):squeeze()

        -- output data
        return {img, torch.LongTensor{classID}}
    end

    return fetch_data, dbloader:size(mode)[1]
end


local function loader_ilsvrc2012(dbloader, num_classes, mode)
    local mode = mode
    if mode == 'test' then
        mode = 'val' -- only has the 'val' set in the ilsvrc2012
    end

    local function fetch_data()
        -- get random class index
        local classID = torch.random(1, num_classes)
        local imgs_class = dbloader:get(mode, 'list_image_filenames_per_class', classID)

        -- remove negative indexes
        imgs_class = imgs_class[imgs_class:ge(0)]

        -- get random image index
        local random_idx = torch.random(1, imgs_class:size(1))
        local img_idx = imgs_class[random_idx] + 1 -- lua is 1-indexed

        -- fetch image filename
        local filename = dbloader:get(mode, 'image_filenames', img_idx)
        filename = ascii2str(filename)[1]
        filename = paths.concat(dbloader.data_dir, filename) -- merge data dirpath to filename

        -- output data
        return {image.load(filename, 3, 'float'), torch.LongTensor{classID}}
    end

    return fetch_data, dbloader:size(mode)[1]
end


local function data_loader(name, mode)
    local dbc = require 'dbcollection.manager'
    local dbdataset = dbc.load(name)
    local num_classes = dbdataset:size('train', 'classes')[1]

    if name == 'ilsvrc2012' then
        return loader_ilsvrc2012(dbdataset, num_classes, mode)
    elseif name == 'cifar10' then
        return loader_cifar(dbdataset, num_classes, mode)
    elseif name == 'cifar100' then
        return loader_cifar(dbdataset, num_classes, mode)
    elseif name == 'mnist' then
        return loader_mnist(dbdataset, num_classes, mode)
    else
        error('Undefined dataset for this example code. Please choose one dataset from the following: ilsvrc2012 | cifar10 | cifar 100 | mnist')
    end
end


return data_loader