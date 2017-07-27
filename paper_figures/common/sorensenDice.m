function qs = sorensenDice(A, B)
% Helper function to calculate Sorensen-Dice score
    a = sum(A(:));
    b = sum(B(:));
    intersect = sum( A(:) .* B(:) );
    qs = 2 * intersect / (a + b);
end

