function plot_2d_visualization(COVlist_unb,class_labels)
feature_set_2d = reshape(COVlist_unb, 72, [])';
mapped_feature_set = tsne(feature_set_2d');
cmap = [0 0 1; 1 0 0];
scatter(mapped_feature_set(:,1), mapped_feature_set(:,2), 10, class_labels,'filled');
colormap(cmap);
% labels={"MCI","Healthy"}
% legend(labels)
end
