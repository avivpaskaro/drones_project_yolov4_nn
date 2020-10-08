%{
Description: This script converts a dataset of videos+GT to 
             collection of jpg photos, and text files which describe 
             the elements in the photos.

Creators: Aviv Paskaro, Stav Yeger

Date: Dec-2019  
%}

if (~exist('TarinRepository', 'dir'))
    mkdir('TarinRepository')
end

dir_name = dir('.');
for ii = 3:length(dir_name)
    if(contains(dir_name(ii).name, 'DataSet'))
        gt_dir = dir([dir_name(ii).folder, dir_name(ii).name, '\GT\']); 
        drone_dir = dir([dir_name(ii).folder, dir_name(ii).name, '\Video\']);
        for jj = 3:length(gt_dir) 
            % Find the last save file index for not override in save
            dir_dataset = dir('TarinRepository');
            dataset_length = 0;
            if(length(dir_dataset) > 2)   
                dataset_length = sscanf(dir_dataset(end).name,'%d');
            end

            v_gt = VideoReader([gt_dir(jj).folder, '\', gt_dir(jj).name]);
            v_drone = VideoReader([drone_dir(jj).folder, '\', drone_dir(jj).name]);
            steps = int16(v_gt.Duration * v_gt.FrameRate);
            start_t = tic;
            step = 1;
            load_bar  = waitbar(0,'Please wait...');

            while hasFrame(v_gt) && hasFrame(v_drone)
                % read video and clear one frame from saving noise
                frame_drone = readFrame(v_drone);
                frame_gt = readFrame(v_gt);
                frame_gt = bwareafilt(frame_gt(:,:,1)>30, 3);
                frame_gt = bwpropfilt(frame_gt,'Area',[30 1000000000]);
                
                % find bounding box from the ground truth
                CC = bwconncomp(frame_gt);
                rows  = size(frame_gt,1);
                cols  = size(frame_gt,2);
                bounding_box = regionprops(CC,'BoundingBox');
                centroid = regionprops(CC,'Centroid');
                if(~isempty(bounding_box))
                    bb = vertcat(bounding_box.BoundingBox);
                    cen = vertcat(centroid.Centroid);
                    width = bb(:,3)/cols;
                    height = bb(:,4)/rows;       
                    center_x  = cen(:,1)/cols;
                    center_y = cen(:,2)/rows;
                    y_hat = zeros(length(center_x), 1);
                    T = cat(2, y_hat, center_x, center_y, width, height);
                else
                    T = [];
                end

                % status bar
                t        = toc(start_t);
                rem_time = (t/step)*(steps-step);
                m        = floor(rem_time/60);
                s        = round(rem_time-m*60);
                prog_str = sprintf('Progress: %2.1f%%  Time Remain:%2.0f:%2.0f', double(step)/double(steps)*100, m, s);
                waitbar(double(step)/double(steps), load_bar, prog_str);   

                % writing
                imwrite(frame_drone,['TarinRepository\', sprintf('%08d', dataset_length+step), '.jpg']);
                writematrix(T, ['TarinRepository\', sprintf('%08d', dataset_length+step), '.txt'], 'Delimiter', 'space')
                step = step + 1;
            end
            close(load_bar);
        end
   end
end  