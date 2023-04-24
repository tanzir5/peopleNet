
'''
Common feature names

Feature                           |  name in dataset
mother's race or ethnicity       ->      cm1ethrace
mother's level of education      ->      cm1edu
mother's marital status          ->      cm1relf
'''


'''
proxy features:
gpa: t5c13a, t5c13b, t5c13c
grit: t5b2b, t5b4y, t5b4z
material_hardship: 
eviction: m5f23d, f5f23d, n5g1d
layoff: m5i4, f5i4
job_training: f5i3b, m5i3b
'''

'''
Common feature descriptions
cm1ethrace                                    Mother race (baseline own report)
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  race_mw1

                 range:  [-9,4]                       units:  1
         unique values:  6                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                             1        -9  -9 Not in wave
                             5        -3  -3 Missing
                           933         1  1 white, non-hispanic
                         2,090         2  2 black, non-hispanic
                         1,050         3  3 hispanic
                           163         4  4 other

-------------------------------------------------------------------------------
cm1edu                                   Mother baseline education (own report)
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  educ_mw1

                 range:  [-9,4]                       units:  1
         unique values:  6                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                             1        -9  -9 Not in wave
                             3        -3  -3 Missing
                         1,402         1  1 less hs
                         1,307         2  2 hs or equiv
                         1,064         3  3 some coll, tech
                           465         4  4 coll or grad

-------------------------------------------------------------------------------
cm1relf                               Constructed-Household Relationship-mother
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  cm1hhrel_mw1

                 range:  [-9,7]                       units:  1
         unique values:  9                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                             1        -9  -9 Not in wave
                             1        -3  -3 Missing
                         1,030         1  1 married
                         1,520         2  2 cohab
                         1,130         3  3 visiting
                           255         4  4 friends
                           159         5  5 hardly talk
                           120         6  6 never talk
                            26         7  7 fath unknwn

-------------------------------------------------------------------------------

'''


'''
GPA desc:
-------------------------------------------------------------------------------
t5c13a                               C13A. Child's language and literacy skills
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  TE_2F

                 range:  [-9,5]                       units:  1
         unique values:  9                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                         2,228        -9  -9 Not in wave
                             1        -3  -3 Missing
                             7        -2  -2 Don't know
                             6        -1  -1 Refuse
                           193         1  1 far below average
                           531         2  2 below average
                           728         3  3 average
                           441         4  4 above average
                           107         5  5 far above average

-------------------------------------------------------------------------------
t5c13b                                 C13B. Child's science and social studies
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  TE_2F

                 range:  [-9,5]                       units:  1
         unique values:  9                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                         2,228        -9  -9 Not in wave
                             1        -3  -3 Missing
                             6        -2  -2 Don't know
                            10        -1  -1 Refuse
                           125         1  1 far below average
                           364         2  2 below average
                         1,057         3  3 average
                           393         4  4 above average
                            58         5  5 far above average

-------------------------------------------------------------------------------
t5c13c                                        c13C. Child's mathematical skills
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  TE_2F

                 range:  [-9,5]                       units:  1
         unique values:  9                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                         2,228        -9  -9 Not in wave
                             1        -3  -3 Missing
                             8        -2  -2 Don't know
                             8        -1  -1 Refuse
                           179         1  1 far below average
                           501         2  2 below average
                           779         3  3 average
                           448         4  4 above average
                            90         5  5 far above average
--------------------------------------------------------------------------------
'''

'''
Grit desc
-------------------------------------------------------------------------------
t5b2b                                   B2B. Child persists in completing tasks
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  FREQ

                 range:  [-9,4]                       units:  1
         unique values:  7                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                         2,228        -9  -9 Not in wave
                             1        -2  -2 Don't know
                             6        -1  -1 Refuse
                           153         1  1 Never
                           638         2  2 Sometimes
                           644         3  3 Often
                           572         4  4 Very often
-------------------------------------------------------------------------------
t5b4y                        B4Y. Child fails to finish things he or she starts
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  TRUTH

                 range:  [-9,3]                       units:  1
         unique values:  8                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                         2,228        -9  -9 Not in wave
                             1        -3  -3 Missing
                             2        -2  -2 Don't know
                             6        -1  -1 Refuse
                           960         0  0 Not true
                           600         1  1 Just a little true
                           279         2  2 Pretty much true
                           166         3  3 Very much true

-------------------------------------------------------------------------------
t5b4z                                  B4Z. Child does not follow through on
                                       instructions and fails to finish
                                       homework
-------------------------------------------------------------------------------

                  type:  numeric (byte)
                 label:  TRUTH

                 range:  [-9,3]                       units:  1
         unique values:  8                        missing .:  0/4,242

            tabulation:  Freq.   Numeric  Label
                         2,228        -9  -9 Not in wave
                             1        -3  -3 Missing
                             4        -2  -2 Don't know
                             5        -1  -1 Refuse
                         1,053         0  0 Not true
                           525         1  1 Just a little true
                           263         2  2 Pretty much true
                           163         3  3 Very much true

'''