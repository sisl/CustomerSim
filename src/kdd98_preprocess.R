# Creating transition data from KDD1998 data competition

library("zipcode")
library("plyr")

print("Loading data...")
# loading train data
data_train <- read.csv("../kdd98_data/cup98LRN.txt")

# loading validation data
data_val <- read.csv("../kdd98_data/cup98VAL.txt")
data_trgt <- read.csv("../kdd98_data/valtargt.txt")

# adding back "CONTROLN" reward variable 
# that is provided separately from the validation file
data_val <- merge(data_val,data_trgt,by="CONTROLN")

# making sure the order of columns is the same in train and val data sets
data_val <- data_val[colnames(data_train)]

# inspect colnames
# colnames(data_train)

print("Processing data...")
# temporarily combining train and val datasets into one array 
# for feature extraction
# no information is shared between the 2 - we later split them again
data_main <- rbind(data_train, data_val)

# add customer index
data_main$customer <- 1:dim(data_main)[1]

# GET MAILING DATES
dates <- data_main[,c(362:384)]

# display variable names
# head(data_main[,c(385:407)])

# NOTE: while the dates of each mailing vary, for simplicity we only consider
# that all mailings were sent sequentially to each individual, assuming that sequences are equally spaced
# in reality, a period from less than one month to 2 months may have passed between 2 mailings in a sequence
# however, the information is not granular enough to know it precisely

# GET MARKETING INTERACTION FREQUENCY PER PERSON
# we assume interaction happened if data is not missing,
# and interaction didn't happen if data is missing
# data providers were not clear about why exactly data is missing,
# but we find that interaction frequency derived this way is helpful for targeting customers

interactions <- dates
interactions[!is.na(interactions)] <- 1
interactions[is.na(interactions)] <- 0
# deriving interaction frequency
# we operate on columns - where dates progress (IMPORTANT!) from right to left
interact_freq <- dates
interact_freq[] <- 0

n <- dim(interact_freq)[2]
for (i in 1:n) { 
  if (1 + n - i < n) {
    interact_freq[,1+n-i] <- interactions[,2+n-i] + interact_freq[,2+n-i]
  }
}

# GET MARKETING INTERACTION RECENCY
interact_recen <- dates
interact_recen[] <- 0
n <- dim(interact_recen)[2]
for (i in 1:n) {
  if (1+n-i < n) {
    interact_recen[,1+n-i] <- (interact_recen[,2+n-i] + 1) * (interactions[,2+n-i]==0) 
  }
}

# GET PRIMARY RFM DATA
data <- data_main[,c(472,435:456)]
data[,1][data[,1]==0] <- NA

# GET FREQUENCY OF DONATIONS
# similar process as in marketing interaction frequency above
transactions <- data
transactions[!is.na(transactions)] <- 1
transactions[is.na(transactions)] <- 0

freq <- data
freq[] <- 0
n <- dim(freq)[2]
for (i in 1:n) {
  if (1+n-i < n) {
    freq[,1+n-i] <- freq[,2+n-i] + transactions[,2+n-i]
  }
}

# GET RECENCY OF DONATIONS
# similar process as in recency of marketing interactions above
recen <- data
recen[] <- 0
n <- dim(recen)[2]
for (i in 1:n) {
  if (1+n-i < n) {
    recen[,1+n-i] <- (recen[,2+n-i]+1)*(transactions[,2+n-i]==0)
  }
}

# GET REWARD DATA
reward <- data
reward[is.na(reward)] <- 0

# GET AVG MONETARY VALUE OF DONATIONS (not counting zeros)
monet <- data
monet[] <- 0
n <- dim(monet)[2]
for (i in 1:n) {
  if (1+n-i < n) {
      monet[,1+n-i] <- (monet[,2+n-i]*freq[,(2+n-i)] + reward[,(2+n-i)])/(freq[,1+n-i]+1*(freq[,1+n-i]==0))
  }
}

# GET ACTIONS (INCLUDING INACTION WHEN NO MAILING HAPPENED)
# we assign the mailing type to each mailing - based on data dictionary 
action_set <- c(5,5,7,6,1,11,8,3,2,10,9,4,5,7,1,11,8,3,2,10,9,4,5)
actions <- matrix(rep(action_set,dim(dates)[1]), nrow = dim(dates)[1], ncol = length(action_set), byrow=TRUE)
# when data is missing, we assign a special type of action - action 0
# we assume missing data to imply no mailing was sent, thus action 0 can be interpreted as inaction
actions[is.na(dates)] <- 0

# ZIP
# colnames(data_main)
zip <- data_main$ZIP
zip <- gsub("[^0-9]", "", zip)
data(zipcode)
zip <- data.frame(zip)
zip <- join(zip, zipcode, by = "zip")
zip$region <- as.numeric(substr(zip$zip, 1, 1)) + 1
zip[is.na(zip)] <- 0
zip_region <- as.matrix(zip$region)
zip_la <- as.matrix(zip$latitude)
zip_lo <- as.matrix(zip$longitude)

# AGE
age <- as.matrix(data_main$AGE)
age[is.na(age)] <- 0
age[age<18] <- 0

# GENDER
gender <- as.matrix(as.numeric(data_main$GENDER))
gender[gender == 1 | gender == 2 | gender == 3 | gender == 5 | gender == 7] <- 3 # other
gender[gender == 4] <- 2 # female
gender[gender == 6] <- 1 # male
gender <- gender - 1

# INCOME
income <- as.matrix(as.numeric(data_main$INCOME))
income[is.na(income)] <- 0

# CUSTOMER
customer <- as.matrix(as.numeric(data_main$customer))

print("Saving data...")

# function to save the data - without and with discretization
save_data <- function(recen, freq, monet, interact_recen,
                      interact_freq, age, gender, income, 
                      zip_region, zip_la, zip_lo, customer, filename) {
  
  for (i in 1:(dim(recen)[2]-1)) {

    r0 <- recen[,1+dim(recen)[2]-i]
    r1 <- recen[,dim(recen)[2]-i]
    f0 <- freq[,1+dim(freq)[2]-i]
    f1 <- freq[,dim(freq)[2]-i]
    m0 <- monet[,1+dim(monet)[2]-i]
    m1 <- monet[,dim(monet)[2]-i]
    ir0 <- interact_recen[,1+dim(monet)[2]-i]
    ir1 <- interact_recen[,dim(monet)[2]-i]
    if0 <- interact_freq[,1+dim(monet)[2]-i]
    if1 <- interact_freq[,dim(monet)[2]-i]
    
    a <- actions[,1+dim(recen)[2]-i]
    rew <- reward[,1+dim(recen)[2]-i]

    period <- rep(i,length(age))
    
    if (i==1) {
      write.table(data.frame(cbind(customer,period,r0,f0,m0,ir0,if0,gender,age,income,zip_region,zip_la,zip_lo,a,rew,r1,f1,m1,ir1,if1,gender,age,income,zip_region,zip_la,zip_lo)),filename,col.names=FALSE,sep=",",row.names=FALSE,eol="\r\n")
    } else {
      write.table(data.frame(cbind(customer,period,r0,f0,m0,ir0,if0,gender,age,income,zip_region,zip_la,zip_lo,a,rew,r1,f1,m1,ir1,if1,gender,age,income,zip_region,zip_la,zip_lo)),filename,col.names=FALSE,sep=",",append=TRUE,row.names=FALSE,eol="\r\n")
    }
  }
}

save_data(recen, freq, monet, interact_recen, interact_freq, 
          age,gender, income, zip_region, zip_la, zip_lo, customer,
          "../kdd98_data/kdd1998tuples.csv")

print("Done")
