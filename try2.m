%try again

Data = readtable("spotify_dataset.xlsx");
Data.explicit = double(Data.explicit);



Data = Data( ~strcmp(Data.decade,'2020-2029') | strcmp(Data.decade, '2020-2029') & Data.popularity >= 63, :);
result  = Data(:, 'popularity');
Y1 = table2array(result);
subset11 = Data(:, {'decade'});
subset12 = Data(:, {'explicit', 'songLength_ms_', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key','liveness','loudness','speechiness','tempo','timeSignature', 'mode', 'valence'});

subset11 = table2array(subset11);
subset12 = table2array(subset12);


% Convert decades to integers using if-elseif
for i = 1:numel(subset11)
    if subset11{i} == '2020-2029'
        subset11{i} = 12;
    elseif subset11{i} == '2010-2019'
        subset11{i} = 11;
    elseif subset11{i} == '2000-2009'
        subset11{i} = 10;
    elseif subset11{i} == '1990-1999'
        subset11{i} = 9;
    elseif subset11{i} == '1980-1989'
        subset11{i} = 8;
    elseif subset11{i} == '1970-1979'
        subset11{i} = 7;
    elseif subset11{i} == '1960-1969'
        subset11{i} = 6;
    elseif subset11{i} == '1950-1959'
        subset11{i} = 5;
    elseif subset11{i} == '1940-1949'
        subset11{i} = 4;
    elseif subset11{i} == '1930-1939'
        subset11{i} = 3;
    elseif subset11{i} == '1920-1929'
        subset11{i} = 2;
    elseif subset11{i} == '1910-1919'
        subset11{i} = 1;
    elseif subset11{i} == '1900-1909'
        subset11{i} = 0;
    end
end

subset11 = cell2mat(subset11);
X1 = [subset11 subset12];


X1 = X1(~any(isnan(X1), 2), :);
Y1 = Y1(~any(isnan(X1), 2), :);
X1 = X1(Y1~=0, :);
Y1 = Y1(Y1~=0);

size(Y1);
variableNames = subset1.Properties.VariableNames;

scatter(X1(:,1), Y1)

csv = [X1 Y1];
T=array2table(csv, 'VariableNames', {'decade', 'explicit', 'songLength_ms_', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'timeSignature', 'mode', 'valence', 'popularity'})

writetable(T, 'preprocessed.csv');

hold on
figure
histogram(Y1)
xline(8, '-m');
xline(16, '-m')
xline(31, '-m')
xline(54, '-m')
xline(61, '-m')
xline(66, '-m')
xline(70, '-m')
xline(75, '-m')
xline(79, '-m')
hold off