% Adapted by: Oscar Herrera G.
% inputs: y, pmax, crit = {'AIC', 'BIC}

% (Official Doc)

% ============================================================
% Perform and display lag order selection tests for AR(p) model, i.e.
% Akaike, Schwartz and Hannan-Quinn information criteria
% ============================================================

% ============================================================

% INPUTS
%   - y     : data vector [periods x 1]
%   - pmax  : number of maximum lags to consider. [scalar]
%   - crit  : criteria to compute lag order selection;
%             possible values: 'AIC', 'SIC', 'HQC'

% OUTPUTS
%   - nlag  : number of lags recommended by the selected information criterion
% ============================================================

% Willi Mutschler, November 2, 2021
% willi@mutschler.eu
% ============================================================


function nlag = LagOrderSelectionARp(y,pmax,crit)

    T = size(y,1);
    T_eff = T - pmax;  % Effective sample size used for all estimations, i.e.,
                       % number of presample values set aside for estimation
                       % isdetermined by the maximum order pmax
    INFO_CRIT = nan(pmax,1);

    % Construct regressor matrix and dependent variable
    % number of presample values set aside is determined by pmax

    Y = lagmatrix(y,1:pmax);
    y = y((pmax+1):T);
    YMAX = Y((pmax+1):T,:);
    YMAX = [ones(T_eff, 1) YMAX];

    for p=1:pmax
        n = 1+p;
        Y = YMAX(:,1:n);
        thetahat = (Y'*Y)\(Y'*y);
        uhat = y - Y*thetahat;
        sigmau2 = uhat'*uhat/T_eff;  % ML estimate of variance errors
        if strcmp(crit, 'AIC')
            INFO_CRIT(p) = log(sigmau2) + 2/T_eff*n;
        elseif strcmp(crit, 'BIC')
            INFO_CRIT(p) = log(sigmau2) + log(T_eff)/T_eff*n;
        end
    end

    % Store results and find minimum value of crit
    nlag = find(INFO_CRIT == min(INFO_CRIT));


end

